#include "ParallelTraining.h"
#include <sstream>

#define SUCCESS 0
#define FAILURE 1

#define NUM_UNIQUE_VALS (NORM_FACTOR + 1) * BPT_NUM_FEATURES;

void checkError(int error, std::string message);

cl::Kernel* kernel(cl::Context context, cl::Device device, string file) {
	std::stringstream ss;
	std::string s;

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
	printf("Kernel name: %s. Program built successfully.\n", kernelName.c_str());

	cl::Kernel* k = new cl::Kernel(program, kernelName.c_str(), &err);

	checkError(err, "Kernel creation failed!");

	return k;
}

int trainingsLoop() {

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

	cl::vector<cl::size_type> sizes = default_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: >";
	for (cl::size_type s : sizes) {
		std::cout << s << ",";
	}
	std::cout << std::endl;

	std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE" << default_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

	cl::Context context({ default_device });

	////////----------------------------------
	//	Create Buffer here
	////////----------------------------------

	short datasetLength = (short)(numFeatures() + 1);
	short datasetValues = datasetLength * numDatasets();
	std::cout << "Dataset Length: >" << datasetLength << "<, Dataset Values: >" << datasetValues << "<" << std::endl;
	unsigned short* dataset = new unsigned short[datasetValues];

	cl::Buffer impurity_buffer_1(context, CL_MEM_READ_WRITE, sizeof(unsigned int) * NUM_CATEGORIES, NULL, &err);
	checkError(err, "impurity_buffer_1 <could not be created");
	cl::Buffer impurity_buffer_2(context, CL_MEM_READ_WRITE, sizeof(unsigned int) * NUM_CATEGORIES, NULL, &err);
	checkError(err, "impurity_buffer_2 could not be created");
	cl::Buffer dataset_buffer(context, CL_MEM_READ_ONLY, sizeof(unsigned short) * datasetValues, NULL, &err);
	checkError(err, "dataset_buffer could not be created");
	cl::Buffer unique_vals_buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * 1001 * numFeatures(), NULL, &err);
	checkError(err, "unique_vals_buffer could not be created");
	/*cl::Buffer split_buffer_1(context, CL_MEM_READ_WRITE, sizeof(unsigned short) * numDatasets(), NULL, &err);
	checkError(err, "split_buffer_1 could not be created");
	cl::Buffer split_buffer_2(context, CL_MEM_READ_WRITE, sizeof(unsigned short) * numDatasets(), NULL, &err);
	checkError(err, "split_buffer_2 could not be created");*/
	// best split (= 2 values), current uncertainty and best information gain
	cl::Buffer temp_values(context, CL_MEM_READ_WRITE, sizeof(double) * 4, NULL, &err);
	checkError(err, "temp_values could not be created");

	short offset = 0;

	for (int i = 0; i < numTrData; i++, offset += datasetLength) {
		Dataset set = trData[i];
		unsigned short* arr = set.toArray();
		memcpy(dataset + offset, arr, datasetLength * sizeof(unsigned short));
	}

	printf("Content of dataset:\n");
	for (int i = 0; i < numDatasets(); i++) {
		for (int j = 0; j < datasetLength; j++) {
			printf("%d ", dataset[i * datasetLength + j]);
		}	
		printf("\n");
	}

	cl::CommandQueue queue(context, default_device, NULL, &err);
	checkError(err, "CommandQueue creation failed!");

	err = queue.enqueueWriteBuffer(dataset_buffer, CL_TRUE, 0, sizeof(unsigned short) * datasetValues, dataset);
	checkError(err, "writing dataset to buffer failed.");

	delete[] dataset;

	////////----------------------------------
	//	Create Kernel here
	////////----------------------------------

	cl::Kernel* kernel_calc_impurity = kernel(context, default_device, "CalcImpurity.cl");
	//std::cout << "CL_KERNEL_WORK_GROUP_SIZE: " << kernel_calc_impurity->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(default_device) << std::endl;
	kernel_calc_impurity->setArg(0, dataset_buffer);
	kernel_calc_impurity->setArg(1, impurity_buffer_1);
	kernel_calc_impurity->setArg(2, temp_values);

	cl::Kernel* kernel_calc_unique_vals = kernel(context, default_device, "CalcUniqueVals.cl");
	kernel_calc_unique_vals->setArg(0, dataset_buffer);
	kernel_calc_unique_vals->setArg(1, unique_vals_buffer);

	////////----------------------------------
	//	Handle Kernels
	////////----------------------------------

	err = queue.enqueueNDRangeKernel(*kernel_calc_impurity, cl::NullRange, cl::NDRange(numDatasets()), cl::NDRange(3));
	checkError(err, "enqueueing kernel_calc_impurity failed.");

	////////----------------------------------
	//	Read results and figure out how to recursively start over on subsets
	////////----------------------------------

	unsigned int imp[NUM_CATEGORIES];
	err = queue.enqueueReadBuffer(impurity_buffer_1, CL_TRUE, 0, sizeof(unsigned int) * NUM_CATEGORIES, imp);
	checkError(err, "enqueueReadBuffer impurity_buffer_1 unsuccessful.");

	double impurity = 1;
	for (int i = 0; i < NUM_CATEGORIES; i++) {
		impurity -= pow(imp[i] / (double) numDatasets(), 2);
	}

	printf("CPU calculated Impurity: %lf\n", impurity);

	double temp[4];
	err = queue.enqueueReadBuffer(temp_values, CL_TRUE, 0, sizeof(double) * 4, temp);
	checkError(err, "enqueueReadBuffer temp_values unsuccessful.");

	if (std::abs(impurity - temp[2]) > 0.0001) {
		std::cerr << "GPU and CPU calculated impurity differ more than 0.0001! -> abort program." << std::endl;
		exit(1);
	}

	////////----------------------------------
	//	Handle Kernels
	////////----------------------------------

	err = queue.enqueueNDRangeKernel(*kernel_calc_unique_vals, cl::NullRange, cl::NDRange(datasetValues));
	checkError(err, "enqueueing kernel_calc_unique_vals failed.");

	unsigned char uniqueVals[10010];
	err = queue.enqueueReadBuffer(unique_vals_buffer, CL_TRUE, 0, sizeof(unsigned char) * 1001 * numFeatures(), uniqueVals);
	checkError(err, "reading unique_vals_buffer failed.");
	const int NUM_VALS_PER_FEATURE = NORM_FACTOR + 1;

	int numUniqueVals = 0;
	std::vector<short> values;
	for (short f = 0; f < numFeatures(); f++) {
		for (short ind = 0; ind <= NORM_FACTOR; ind++) {
			if (uniqueVals[f * NUM_VALS_PER_FEATURE + ind] == 1) {
				//printf("Feature %d - unique Val: %d\n", f, ind);
				numUniqueVals++;
				values.push_back(f);
				values.push_back(ind);
			}
		}
	}

	unsigned short* splitInfo = new unsigned short[numUniqueVals * 2];
	for (int i = 0; i < numUniqueVals * 2; i++) {
		splitInfo[i] = values.at(i);
	}

	////////----------------------------------
	//	Handle Kernels
	////////----------------------------------
	cl::Buffer split_info_buffer(context, CL_MEM_READ_ONLY, sizeof(unsigned short) * numUniqueVals * 2, NULL, &err);
	checkError(err, "unique_vals_buffer could not be created");

	cl::Buffer read_split_info_buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned short) * numUniqueVals * 16 * 2, NULL, &err);
	checkError(err, "read_split_info_buffer could not be created");

	cl::Buffer split_info_gain_buffer(context, CL_MEM_READ_WRITE, sizeof(double) * numUniqueVals, NULL, &err);
	checkError(err, "split_info_gain_buffer could not be created");
	
	cl::Buffer category_buffer(context, CL_MEM_READ_WRITE, sizeof(unsigned short) * 12, NULL, &err);
	checkError(err, "category_buffer could not be created");

	err = queue.enqueueWriteBuffer(split_info_buffer, CL_TRUE, 0, sizeof(unsigned short) * numUniqueVals * 2, splitInfo);
	checkError(err, "writing splitInfo to buffer failed.");

	cl::Kernel* kernel_partition_data = kernel(context, default_device, "PartitionData.cl");
	kernel_partition_data->setArg(0, dataset_buffer);
	kernel_partition_data->setArg(1, temp_values);
	kernel_partition_data->setArg(2, split_info_buffer);
	kernel_partition_data->setArg(3, read_split_info_buffer);
	kernel_partition_data->setArg(4, split_info_gain_buffer);
	kernel_partition_data->setArg(5, category_buffer);

	int work_item_per_group = std::min(1024, numDatasets());

	err = queue.enqueueNDRangeKernel(*kernel_partition_data, cl::NullRange, cl::NDRange(numUniqueVals * work_item_per_group), cl::NDRange(work_item_per_group));
	checkError(err, "enqueueing kernel_partition_data failed.");
	queue.finish();

	unsigned short* readSplit = new unsigned short[numUniqueVals * 16 * 2];
	err = queue.enqueueReadBuffer(read_split_info_buffer, CL_TRUE, 0, sizeof(unsigned short) * numUniqueVals * 2 * 16, readSplit);
	checkError(err, "reading read_split_info_buffer failed.");

	offset = 32;
	int result = 0;
	for (int i = 0; i < numUniqueVals; i++) {
		int sum = readSplit[i*offset] + readSplit[i*offset + 16];
		if (sum != 12 && sum != 0) {	// 0 for skipped kernels, because split resulted in branch with size 0!
			printf("FAILED: ");
		}
		else
			continue;
		printf("Group ID %d calculated: >", i);
		for (int j = 0; j < 15; j++) {
			printf("%d,", readSplit[i*offset + j]);
		}
		printf("%d< and >", readSplit[i*offset + 15]);

		for (int j = 0; j < 15; j++) {
			printf("%d,", readSplit[i*offset + j + 16]);
		}
		printf("%d<\n", readSplit[i*offset + 15 + 16]);
	}

	double* readInfoGain = new double[numUniqueVals];
	err = queue.enqueueReadBuffer(split_info_gain_buffer, CL_TRUE, 0, sizeof(double) * numUniqueVals, readInfoGain);
	checkError(err, "reading split_info_gain_buffer failed.");

	for (int i = 0; i < numUniqueVals; i++) {
		printf("Split on Feature %d with value %d -> Info gain = %lf\n", splitInfo[i * 2], splitInfo[i * 2 + 1], readInfoGain[i]);
	}

	unsigned short categories[12];
	err = queue.enqueueReadBuffer(category_buffer, CL_TRUE, 0, sizeof(unsigned short) * 12, categories);
	checkError(err, "reading category_buffer failed");

	delete[] splitInfo;
	delete[] readInfoGain;

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