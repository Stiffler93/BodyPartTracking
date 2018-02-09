#include "ParallelTraining.h"
#include "TreeUtils.h"
#include <sstream>
#include <omp.h>
#include <fstream>
#include <map>
#include <iostream>

#define SUCCESS 0
#define FAILURE 1

#define KERNEL_CALC_IMPURITY_1 0
#define KERNEL_CALC_IMPURITY_2 1
#define KERNEL_CALC_UNIQUE_VALS 2
#define KERNEL_FIND_BEST_SPLIT 3

using namespace tree;
using std::string;

static int infoGainFailures = 0;

void checkError(int error, std::string message);

cl::Kernel* kernel(cl::Context context, cl::Device device, string file, std::map<string, string> macros = std::map<string,string>()) {
	std::stringstream ss;
	std::string s;

	ss << "#define DS_LEN " << (numFeatures() + 1) << std::endl;
	ss << "#define NUM_CATEGORIES " << NUM_CATEGORIES << std::endl;
	ss << "#define NUM_FEATURES " << numFeatures() << std::endl;
	ss << "#define CATEGORY " << numFeatures() << std::endl;
	for (std::map<string, string>::iterator it = macros.begin(); it != macros.end(); ++it) {
		ss << "#define " << it->first << " " << it->second << std::endl;
	}

	std::ifstream readKernel(file);
	while (std::getline(readKernel, s)) {
		ss << s << std::endl;
	}

	readKernel.close();
	
	cl::Program::Sources sources;
	std::string kernel_code = ss.str();

	sources.push_back({ kernel_code.c_str(),kernel_code.length() });
	cl::Program program(context, sources);

	cl_int err;
	if ((err = program.build({ device })) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		std::cout << "Error Code: " << err << std::endl;
		std::cin >> s;
		exit(1);
	}

	std::string kernelName;
	size_t start = file.find_last_of('/');
	size_t endf = file.find_last_of('.');

	if (start == std::string::npos)
		start = 0;

	kernelName = file.substr(start, endf - start);

	cl::Kernel* k = new cl::Kernel(program, kernelName.c_str(), &err);
	checkError(err, "Kernel creation failed!");

	return k;
}

void trainingsLoop(Record * trData, const unsigned int numTrData, Node *& node, cl::Context& context, cl::Device& device, cl::CommandQueue& queue, unsigned long* numTrDataLeft, cl::Kernel* kernels, int cycle = 1) {
	//std::printf("Call trainingsLoop(). numTrData = %d\n", numTrData);
	std::printf("Tree Depth: %4d. numTrDataLeft: %10ld\r", cycle++, *numTrDataLeft);
	
	bool firstTime = true;

	BestSplit split;
	double imp = 1;

	// changing from parallel to linear algorithm causes huge errors!
	if (false && numTrData < 1600 && numTrData > BPT_STOP_EVALUATION_LIMIT) { // CPU (is faster for small data
		trace("CPU calculation");
		imp = impurity(trData, numTrData);

		BestSplit split;
		if (imp > BPT_STOP_EVALUATION_IMPURITY)
			split = findBestSplit(trData, numTrData);
	} 
	else if(numTrData > BPT_STOP_EVALUATION_LIMIT) {	// calculation on GPU
		trace("GPU calculation");
		cl_int err = 0;
		const int NUM_VALS_PER_FEATURE = NORM_FACTOR + 1;
		const unsigned int NUM_COMPUTE_UNITS = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

		short datasetLength = (numFeatures() + 1);
		unsigned long datasetValues = datasetLength * numTrData;
		unsigned short* dataset = new unsigned short[datasetValues]; //1.331.680 Byte throws: C++-Ausnahme: std::bad_array_new_length

		cl::Buffer* impurity_buffer = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int) * NUM_CATEGORIES, NULL, &err);
		checkError(err, "impurity_buffer <could not be created");
		cl::Buffer* dataset_buffer = new cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(unsigned short) * datasetValues, NULL, &err);
		checkError(err, "dataset_buffer could not be created");
		// best split (= 2 values), current uncertainty and best information gain
		cl::Buffer* temp_values_buffer = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * 4, NULL, &err);
		checkError(err, "temp_values could not be created");

		trace("Create Dataset");

		Record set;
		for (unsigned int i = 0, offset = 0; i < numTrData; i++, offset += datasetLength) {
			set = trData[i];
			unsigned short* arr = set.toArray();
			memcpy(dataset + offset, arr, datasetLength * sizeof(unsigned short));
		}

		err = queue.enqueueWriteBuffer(*dataset_buffer, CL_TRUE, 0, sizeof(unsigned short) * datasetValues, dataset);
		checkError(err, "writing dataset to buffer failed.");

		delete[] dataset;

		trace("Wrote dataset to buffer");

		unsigned int* imp_buffer = new unsigned int[NUM_CATEGORIES];
		for (int i = 0; i < NUM_CATEGORIES; i++)
			imp_buffer[i] = 0;

		err = queue.enqueueWriteBuffer(*impurity_buffer, CL_TRUE, 0, sizeof(unsigned int) * NUM_CATEGORIES, imp_buffer);
		checkError(err, "writing imp_buffer to buffer failed.");

		trace("Initialized impurity buffer");

		delete[] imp_buffer;

		cl::Kernel kernel_calc_impurity = kernels[KERNEL_CALC_IMPURITY_1];
		kernel_calc_impurity.setArg(0, *dataset_buffer);
		kernel_calc_impurity.setArg(1, *impurity_buffer);
		kernel_calc_impurity.setArg(2, numTrData);

		int tenthOfNumTrData = (int)ceil((double)numTrData / (double)NUM_COMPUTE_UNITS);
		int work_item_per_group = std::min(256, tenthOfNumTrData);
		int multiplicator = (int)ceil((double)numTrData / (double)work_item_per_group);
		err = queue.enqueueNDRangeKernel(kernel_calc_impurity, cl::NullRange, cl::NDRange(work_item_per_group * multiplicator), cl::NDRange(work_item_per_group));
		checkError(err, "enqueueing kernel_calc_impurity failed.");

		queue.finish();

		cl::Kernel kernel_calc_impurity2 = kernels[KERNEL_CALC_IMPURITY_2];
		kernel_calc_impurity2.setArg(0, *impurity_buffer);
		kernel_calc_impurity2.setArg(1, *temp_values_buffer);
		kernel_calc_impurity2.setArg(2, numTrData);

		err = queue.enqueueNDRangeKernel(kernel_calc_impurity2, cl::NullRange, cl::NDRange(NUM_CATEGORIES), cl::NDRange(NUM_CATEGORIES));
		checkError(err, "enqueueing kernel_calc_impurity2 failed.");

		queue.finish();

		trace("Kernel -Impurity- finished");

		err = queue.enqueueReadBuffer(*temp_values_buffer, CL_TRUE, 2 * sizeof(double), sizeof(double), &imp);
		checkError(err, "reading temp_values_buffer impurity failed!");

		if (imp < 0 || imp > 1) {
			std::printf("Failure. DEBUG mode!\n");
			std::printf("Impurity >%lf< is wrong! Exit program!\n", imp);
		}

		if (imp > BPT_STOP_EVALUATION_IMPURITY) {
			trace("Calc unique Vals");

			cl::Buffer* unique_vals_buffer = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * NUM_VALS_PER_FEATURE * numFeatures(), NULL, &err);
			checkError(err, "unique_vals_buffer could not be created");

			cl::Kernel kernel_calc_unique_vals = kernels[KERNEL_CALC_UNIQUE_VALS];
			kernel_calc_unique_vals.setArg(0, *dataset_buffer);
			kernel_calc_unique_vals.setArg(1, *unique_vals_buffer);
			kernel_calc_unique_vals.setArg(2, numTrData);

			unsigned char* empty_buffer = new unsigned char[NUM_VALS_PER_FEATURE * numFeatures()];
			for (int i = 0; i < NUM_VALS_PER_FEATURE * numFeatures(); i++) {
				empty_buffer[i] = 0;
			}
			err = queue.enqueueWriteBuffer(*unique_vals_buffer, CL_TRUE, 0, sizeof(unsigned char) * NUM_VALS_PER_FEATURE * numFeatures(), empty_buffer);
			delete[] empty_buffer;

			int locSize = std::min(256, (int) numTrData / numFeatures() + 1);
			int gloSize = locSize * numFeatures();
			
			err = queue.enqueueNDRangeKernel(kernel_calc_unique_vals, cl::NullRange, cl::NDRange(gloSize), cl::NDRange(locSize));
			checkError(err, "enqueueing kernel_calc_unique_vals failed.");

			queue.finish();

			trace("unique Vals calculated.");

			unsigned char* uniqueVals = new unsigned char[numFeatures() * NUM_VALS_PER_FEATURE];
			trace("Read Buffer");
			err = queue.enqueueReadBuffer(*unique_vals_buffer, CL_FALSE, 0, sizeof(unsigned char) * NUM_VALS_PER_FEATURE * numFeatures(), uniqueVals);
			trace("Done");
			checkError(err, "reading unique_vals_buffer failed.");

			// changed CL_TRUE to CL_FALSE and added queue.finish() to get error return code!
			queue.finish();
			trace("unique Vals buffer read.");

			delete unique_vals_buffer;

			srand(time(0));
			double randNum = 0;

			int numUniqueVals = 0;
			std::vector<short> values;
			for (short f = 0; f < numFeatures(); f++) {
				for (short ind = 0; ind < NUM_VALS_PER_FEATURE; ind++) {
					//trace("UniqueVal: >" + std::to_string(f) + "," + std::to_string(ind) + "< = " + std::to_string(uniqueVals[f * NUM_VALS_PER_FEATURE + ind]));
					if (uniqueVals[f * NUM_VALS_PER_FEATURE + ind] != 0) {
						//randNum = (double)rand() / (double)RAND_MAX;

						//if (randNum > BPT_UNIQUE_VAL_INVALIDATION_RATE) {
						//	continue;
						//}

						//trace("UniqueVal: >" + std::to_string(f) + "," + std::to_string(ind) + "<");

						numUniqueVals++;
						values.push_back(f);
						values.push_back(ind);
					}
				}
			}

			delete[] uniqueVals;

			unsigned short* splitInfo = new unsigned short[numUniqueVals * 2];
			for (int i = 0; i < numUniqueVals * 2; i += 2) {
				splitInfo[i] = values.at(i);
				splitInfo[i + 1] = values.at(i + 1);
			}

			trace("SplitInfo built");

			cl::Buffer* split_info_buffer = new cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(unsigned short) * numUniqueVals * 2, NULL, &err);
			checkError(err, "split_info_buffer could not be created");

			err = queue.enqueueWriteBuffer(*split_info_buffer, CL_TRUE, 0, sizeof(unsigned short) * numUniqueVals * 2, splitInfo);
			checkError(err, "writing splitInfo to buffer failed.");

			cl::Buffer* split_info_gain_buffer = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * numUniqueVals, NULL, &err);
			checkError(err, "split_info_gain_buffer could not be created");

			trace("Prepare Kernel Best Split");

			cl::Kernel kernel_best_split = kernels[KERNEL_FIND_BEST_SPLIT];
			kernel_best_split.setArg(0, *dataset_buffer);
			kernel_best_split.setArg(1, *temp_values_buffer);
			kernel_best_split.setArg(2, *split_info_buffer);
			kernel_best_split.setArg(3, *split_info_gain_buffer);
			kernel_best_split.setArg(4, numTrData);

			work_item_per_group = std::min((unsigned int) 256, numTrData * 2);

			err = queue.enqueueNDRangeKernel(kernel_best_split, cl::NullRange, cl::NDRange(numUniqueVals * work_item_per_group), cl::NDRange(work_item_per_group));
			checkError(err, "enqueueing kernel_partition_data failed.");

			queue.finish();

			trace("Calculated Best Split");

			delete split_info_buffer;

			double* read_split_info_gain = new double[numUniqueVals];
			err = queue.enqueueReadBuffer(*split_info_gain_buffer, CL_TRUE, 0, sizeof(double) * numUniqueVals, read_split_info_gain);
			checkError(err, "split_info_gain_buffer could not be read.");

			delete split_info_gain_buffer;

			const int FACTOR = 1000000;
			for (int i = 0; i < numUniqueVals; i++) {
				int newGain = (int)(read_split_info_gain[i] * FACTOR);
				int oldGain = (int)(split.gain * FACTOR);
				if (read_split_info_gain[i] < -0.01 || read_split_info_gain[i] > 1) {
					//std::printf("Failure. DEBUG mode!\n");
					infoGainFailures++;
					trace("Impurity = " + std::to_string(imp) + ", Index = " + std::to_string(i) + ", Split Info Gain = " + std::to_string(read_split_info_gain[i])+ " < 0 || > 1 !!!\n");

					if (firstTime) {
						firstTime = false;
						trace("Dataset (wrong info gain!):");
						for (unsigned int i = 0; i < numTrData; i++) {
							trace(trData[i].toString());
						}
					}
				}

				if (newGain > oldGain) {
					split.gain = (float)read_split_info_gain[i];
					split.decision.feature = splitInfo[i * 2];
					split.decision.refVal = splitInfo[i * 2 + 1];
				}
			}

			trace("Found Best Split");

			delete[] read_split_info_gain;
			delete[] splitInfo;
		}

		delete impurity_buffer;
		delete dataset_buffer;
		delete temp_values_buffer;
	}

	if (imp <= BPT_STOP_EVALUATION_IMPURITY || numTrData <= BPT_STOP_EVALUATION_LIMIT || split.gain == 0) {
		trace("Create ResultNode");
		if (numTrData == 1 || imp == 0) {
			Result res;
			res.outcome = trData[0].outcome;
			res.probability = 1.0;
			node = (Node*) new ResultNode(res);
		}
		else {
			std::map<string, int> results;
			for (unsigned int i = 0; i < numTrData; i++) {
				string category = trData[i].outcome;
				std::map<string, int>::iterator val = results.lower_bound(category);

				if (val != results.end() && !(results.key_comp()(category, val->first))) {
					val->second++;
				}
				else {
					results.insert(val, std::map<string, int>::value_type(category, 1));
				}
			}

			int size = (int)results.size();
			int sum = 0;
			for (auto it : results) {
				sum += it.second;
			}

			std::vector<Result> endRes;
			for (auto it : results) {
				Result r;
				r.outcome = it.first;
				r.probability = (float)it.second / (float)sum;
				endRes.push_back(r);
			}

			node = (Node*) new ResultNode(endRes);
		}

		delete[] trData;

		*numTrDataLeft = *numTrDataLeft - numTrData;
		//std::printf("NumTrDataLeft: %ld\n", *numTrDataLeft);

		return;
	}

	trace("Create DecisionNode");

	node = (Node*) new DecisionNode(split.decision);

	Partition part;
	part.true_branch = new Record[numTrData];
	part.false_branch = new Record[numTrData];

	partition(&part, trData, numTrData, split.decision);

	delete[] trData;

	if (part.true_branch_size > 0)
		trainingsLoop(part.true_branch, part.true_branch_size, node->true_branch, context, device, queue, numTrDataLeft, kernels, cycle);

	if (part.false_branch_size > 0)
		trainingsLoop(part.false_branch, part.false_branch_size, node->false_branch, context, device, queue, numTrDataLeft, kernels, cycle);
}

void startParallelTraining(Record * trData, const unsigned int numTrData, Node *& rootNode)
{
	std::printf("Start Parallel Training.\n");
	std::string s;
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	cl_int err;

	for (cl::Platform platform : all_platforms) {
		cl::string name = platform.getInfo<CL_PLATFORM_NAME>(&err);
		std::printf("Platform: %s\n", name.c_str());
	}

	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}

	for (cl::Device device : all_devices) {
		std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	}

	cl::Device default_device = all_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	std::cout << "Local Memory Size: " << default_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;

	cl::Context context({ default_device });

	cl::CommandQueue queue(context, default_device, NULL, &err);
	checkError(err, "CommandQueue creation failed!");

	cl::Kernel kernels[4];

#pragma omp parallel num_threads(4) 
{
	int id = omp_get_thread_num();
	cl::Kernel* k = NULL;

	switch (id) {
	case 0:
		k = kernel(context, default_device, "CalcImpurity.cl");
		break;
	case 1:
		k = kernel(context, default_device, "CalcImpurity2.cl");
		break;
	case 2:
		k = kernel(context, default_device, "CalcUniqueVals.cl");
		break;
	case 3:
		k = kernel(context, default_device, "FindBestSplit.cl");
		break;
	}

	kernels[id] = *k;
	delete k;
}

	std::printf("Kernels created... Start training!\n\n");

	unsigned long numTrDataLeft = numTrData;
	//std::printf("NumTrDataLeft: %ld\n", numTrDataLeft);

	clock_t b = clock();
	trainingsLoop(trData, numTrData, rootNode, context, default_device, queue, &numTrDataLeft, kernels);
	clock_t e = clock();
	std::printf("Tree Depth: %4d. numTrDataLeft: %10ld\n", 0, numTrDataLeft);

	trace("NumInfoGain Calculation Failures: " + std::to_string(infoGainFailures));

	double time = double(e - b) / (double)CLOCKS_PER_SEC;
	printf("Parallel Training took %lf seconds.\n", time);

	std::cout << "Finished." << std::endl;
}

void checkError(int error, std::string message) {
	if (error != 0) {
		std::printf("%s. ErrorCode = %d\n", message.c_str(), error);
		std::string s;
		std::cin >> s;
		exit(1);
	}
}