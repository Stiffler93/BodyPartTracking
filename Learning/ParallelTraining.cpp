#include "ParallelTraining.h"
#include <sstream>
#include <ctime>

#define SUCCESS 0
#define FAILURE 1

#define NUM_UNIQUE_VALS (NORM_FACTOR + 1) * BPT_NUM_FEATURES;

void checkError(int error, std::string message);

cl::Kernel* kernel(cl::Context context, cl::Device device, string file, std::map<string, string> macros = std::map<string,string>()) {
	std::stringstream ss;
	std::string s;

	for (std::map<string, string>::iterator it = macros.begin(); it != macros.end(); ++it) {
		ss << "#define " << it->first << " " << it->second << std::endl;
	}

	ifstream readKernel(file);
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
	size_t end = file.find_last_of('.');

	if (start == std::string::npos)
		start = 0;

	kernelName = file.substr(start, end - start);

	cl::Kernel* k = new cl::Kernel(program, kernelName.c_str(), &err);
	checkError(err, "Kernel creation failed!");

	return k;
}

void trainingsLoop(Dataset * trData, int numTrData, Node *& node, cl::Context& context, cl::Device& device, cl::CommandQueue& queue, unsigned long* numTrDataLeft) {
	//clock_t begin = clock();
	//printf("Call trainingsLoop(). numTrData = %d\n", numTrData);

	BestSplit split;
	double impurity = 1;

	if (numTrData < 2200 && numTrData > BPT_STOP_EVALUATION_LIMIT) { // CPU (is faster for small data
		bool isHeterogenous = false;
		string temp = trData[0].outcome;
		for (int i = 1; i <= numTrData; i++) {
			if (temp != trData[i - 1].outcome) {
				isHeterogenous = true;
				break;
			}
		}

		if (isHeterogenous) {
			split = findBestSplit(trData, numTrData);
		}

		if (isPure(trData, numTrData)) {
			impurity = 0;
		}
	} 
	else if(numTrData > BPT_STOP_EVALUATION_LIMIT) {	// calculation on GPU
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

		Dataset set;
		for (int i = 0, offset = 0; i < numTrData; i++, offset += datasetLength) {
			set = trData[i];
			unsigned short* arr = set.toArray();
			memcpy(dataset + offset, arr, datasetLength * sizeof(unsigned short));
		}

		err = queue.enqueueWriteBuffer(*dataset_buffer, CL_TRUE, 0, sizeof(unsigned short) * datasetValues, dataset);
		checkError(err, "writing dataset to buffer failed.");

		delete[] dataset;

		unsigned int* imp_buffer = new unsigned int[NUM_CATEGORIES];
		for (int i = 0; i < NUM_CATEGORIES; i++)
			imp_buffer[i] = 0;

		err = queue.enqueueWriteBuffer(*impurity_buffer, CL_TRUE, 0, sizeof(unsigned int) * NUM_CATEGORIES, imp_buffer);
		checkError(err, "writing imp_buffer to buffer failed.");

		delete[] imp_buffer;

		cl::Kernel* kernel_calc_impurity = kernel(context, device, "CalcImpurity.cl");
		kernel_calc_impurity->setArg(0, *dataset_buffer);
		kernel_calc_impurity->setArg(1, *impurity_buffer);
		kernel_calc_impurity->setArg(2, numTrData);

		int tenthOfNumTrData = (int)ceil((double)numTrData / (double)NUM_COMPUTE_UNITS);
		int work_item_per_group = std::min(256, tenthOfNumTrData);
		int multiplicator = (int)ceil((double)numTrData / (double)work_item_per_group);
		err = queue.enqueueNDRangeKernel(*kernel_calc_impurity, cl::NullRange, cl::NDRange(work_item_per_group * multiplicator), cl::NDRange(work_item_per_group));
		checkError(err, "enqueueing kernel_calc_impurity failed.");

		delete kernel_calc_impurity;

		cl::Kernel* kernel_calc_impurity2 = kernel(context, device, "CalcImpurity2.cl");
		kernel_calc_impurity2->setArg(0, *impurity_buffer);
		kernel_calc_impurity2->setArg(1, *temp_values_buffer);
		kernel_calc_impurity2->setArg(2, numTrData);

		err = queue.enqueueNDRangeKernel(*kernel_calc_impurity2, cl::NullRange, cl::NDRange(NUM_CATEGORIES), cl::NDRange(NUM_CATEGORIES));
		checkError(err, "enqueueing kernel_calc_impurity2 failed.");

		delete kernel_calc_impurity2;

		err = queue.enqueueReadBuffer(*temp_values_buffer, CL_TRUE, 2 * sizeof(double), sizeof(double), &impurity);

		if (impurity < 0 || impurity > 1) {
			printf("Impurity >%lf< is wrong! Exit program!\n", impurity);
		}

		if (impurity != 0) {
			cl::Buffer* unique_vals_buffer = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * NUM_VALS_PER_FEATURE * numFeatures(), NULL, &err);
			checkError(err, "unique_vals_buffer could not be created");

			cl::Kernel* kernel_calc_unique_vals = kernel(context, device, "CalcUniqueVals.cl");
			kernel_calc_unique_vals->setArg(0, *dataset_buffer);
			kernel_calc_unique_vals->setArg(1, *unique_vals_buffer);

			unsigned char* empty_buffer = new unsigned char[NUM_VALS_PER_FEATURE * numFeatures()];
			for (int i = 0; i < NUM_VALS_PER_FEATURE * numFeatures(); i++) {
				empty_buffer[i] = 0;
			}
			err = queue.enqueueWriteBuffer(*unique_vals_buffer, CL_TRUE, 0, sizeof(unsigned char) * NUM_VALS_PER_FEATURE * numFeatures(), empty_buffer);
			delete[] empty_buffer;

			err = queue.enqueueNDRangeKernel(*kernel_calc_unique_vals, cl::NullRange, cl::NDRange(datasetValues));
			checkError(err, "enqueueing kernel_calc_unique_vals failed.");

			delete kernel_calc_unique_vals;

			unsigned char* uniqueVals = new unsigned char[numFeatures() * NUM_VALS_PER_FEATURE];
			err = queue.enqueueReadBuffer(*unique_vals_buffer, CL_TRUE, 0, sizeof(unsigned char) * NUM_VALS_PER_FEATURE * numFeatures(), uniqueVals);
			checkError(err, "reading unique_vals_buffer failed.");

			delete unique_vals_buffer;

			int numUniqueVals = 0;
			std::vector<short> values;
			for (short f = 0; f < numFeatures(); f++) {
				for (short ind = 0; ind < NUM_VALS_PER_FEATURE; ind++) {
					if (uniqueVals[f * NUM_VALS_PER_FEATURE + ind] == 1) {
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

			cl::Buffer* split_info_buffer = new cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(unsigned short) * numUniqueVals * 2, NULL, &err);
			checkError(err, "split_info_buffer could not be created");

			err = queue.enqueueWriteBuffer(*split_info_buffer, CL_TRUE, 0, sizeof(unsigned short) * numUniqueVals * 2, splitInfo);
			checkError(err, "writing splitInfo to buffer failed.");

			cl::Buffer* split_info_gain_buffer = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * numUniqueVals, NULL, &err);
			checkError(err, "split_info_gain_buffer could not be created");

			cl::Buffer* read_additional_info_buffer = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * 88, NULL, &err);
			checkError(err, "read_additional_info_buffer could not be created");

			std::map<string, string> macros;
			macros.insert(std::pair<string, string>("NUM_DATASETS", to_string(numTrData)));
			//macros.insert(std::pair<string, string>("NEXT_SQUARE_OF_TWO", to_string(nextSquareOfTwo)));

			cl::Kernel* kernel_best_split = kernel(context, device, "FindBestSplit.cl", macros);
			kernel_best_split->setArg(0, *dataset_buffer);
			kernel_best_split->setArg(1, *temp_values_buffer);
			kernel_best_split->setArg(2, *split_info_buffer);
			kernel_best_split->setArg(3, *split_info_gain_buffer);
			kernel_best_split->setArg(4, *read_additional_info_buffer);

			work_item_per_group = std::min(256, numTrData * 2);

			err = queue.enqueueNDRangeKernel(*kernel_best_split, cl::NullRange, cl::NDRange(numUniqueVals * work_item_per_group), cl::NDRange(work_item_per_group));
			checkError(err, "enqueueing kernel_partition_data failed.");

			delete kernel_best_split;
			delete split_info_buffer;

			double* read_split_info_gain = new double[numUniqueVals];
			err = queue.enqueueReadBuffer(*split_info_gain_buffer, CL_TRUE, 0, sizeof(double) * numUniqueVals, read_split_info_gain);
			checkError(err, "split_info_gain_buffer could not be read.");

			double read_additional_info[88];
			err = queue.enqueueReadBuffer(*read_additional_info_buffer, CL_TRUE, 0, sizeof(double) * 88, read_additional_info);
			checkError(err, "read_additional_info could not be read.");

			delete split_info_gain_buffer;

			const int FACTOR = 1000000;
			for (int i = 0; i < numUniqueVals; i++) {
				int newGain = (int)(read_split_info_gain[i] * FACTOR);
				int oldGain = (int)(split.gain * FACTOR);
				if (read_split_info_gain[i] < -0.0001 || read_split_info_gain[i] > 1) {
					printf("Index = %d, Split Info Gain = %lf < 0 || > 1 !!!\n", i, read_split_info_gain[i]);
				}
				if (/*(float) read_split_info_gain[i] > (float) split.gain*/ newGain > oldGain) {
					split.gain = (float)read_split_info_gain[i];
					split.decision.feature = splitInfo[i * 2];
					split.decision.refVal = splitInfo[i * 2 + 1];
				}
			}

			delete[] read_split_info_gain;
			delete[] splitInfo;

			//printf("Best Info Gain = %lf for Feature %d with Value %d\n", split.gain, split.decision.feature, split.decision.refVal);
		}

		delete impurity_buffer;
		delete dataset_buffer;
		delete temp_values_buffer;
	}

	if (numTrData <= BPT_STOP_EVALUATION_LIMIT || split.gain == 0) {
		if (numTrData == 1 || impurity == 0) {
			Result res;
			res.outcome = trData[0].outcome;
			res.probability = 1.0;
			node = (Node*) new ResultNode(res);
		}
		else {
			map<string, int> results;
			for (int i = 0; i < numTrData; i++) {
				string category = trData[i].outcome;
				map<string, int>::iterator val = results.lower_bound(category);

				if (val != results.end() && !(results.key_comp()(category, val->first))) {
					val->second++;
				}
				else {
					results.insert(val, map<string, int>::value_type(category, 1));
				}
			}

			int size = (int)results.size();
			int sum = 0;
			for (auto it : results) {
				sum += it.second;
			}

			vector<Result> endRes;
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
		printf("NumTrDataLeft: %ld\n", *numTrDataLeft);

		return;
	}

	node = (Node*) new DecisionNode(split.decision);

	Partition part;
	part.true_branch = new Dataset[numTrData];
	part.false_branch = new Dataset[numTrData];

	if (part.true_branch_size == numTrData || part.false_branch_size == numTrData) {
		printf("ERROR when partitioning Dataset! Set wasn't reduced!\n");
	}

	partition(&part, trData, numTrData, split.decision);

	delete[] trData;

	//clock_t end = clock();
	//double elapsed_secs = double(end - begin) / (double)CLOCKS_PER_SEC;
	//printf("Cycle took %lf seconds!\n", elapsed_secs);

	if (part.true_branch_size > 0)
		trainingsLoop(part.true_branch, part.true_branch_size, node->true_branch, context, device, queue, numTrDataLeft);

	if (part.false_branch_size > 0)
		trainingsLoop(part.false_branch, part.false_branch_size, node->false_branch, context, device, queue, numTrDataLeft);
}

void startParallelTraining(Dataset * trData, int numTrData, Node *& rootNode)
{
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
		printf("Platform: %s\n", name.c_str());
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

	unsigned long numTrDataLeft = BPT_NUM_DATASETS;
	printf("NumTrDataLeft: %ld\n", numTrDataLeft);

	trainingsLoop(trData, numTrData, rootNode, context, default_device, queue, &numTrDataLeft);

	std::cout << "Finished." << std::endl;
}

void checkError(int error, std::string message) {
	if (error != 0) {
		printf("%s. ErrorCode = %d\n", message.c_str(), error);
		std::string s;
		std::cin >> s;
		exit(1);
	}
}