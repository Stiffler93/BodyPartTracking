
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

//32768 = max local memory, 8192 is a quarter
#define MAX_LOCAL_MEMORY 8192
#define DS_LEN 11
#define NUM_CATEGORIES 6
#define CALC_FACTOR 10000000000
#define CATEGORY 10
#define OTHER 0x0E
#define PARTITION_ONE 0x80
#define PARTITION_TWO 0x40
#define CATEGORY_PART 0x0F

__kernel void FindBestSplit(__global const unsigned short* dataset, __global double* tempVals, __global const unsigned short* splitInfo, 
			__global double* splitInfoGain, __global double* read_additional_info) {

	int gid = get_global_id(0);
	int wid = get_group_id(0);
	int lid = get_local_id(0);
	int lsi = get_local_size(0);

	local unsigned char partition[MAX_LOCAL_MEMORY];
	local unsigned short flags[MAX_LOCAL_MEMORY];

	local unsigned long impurity_buffer_part1[NUM_CATEGORIES];
	local unsigned long impurity_buffer_part2[NUM_CATEGORIES];

	local unsigned long partition_sizes[2];
	local unsigned long cur_uncertainty[2]; // work with factor billion

	unsigned short feat = splitInfo[wid*2];
	unsigned short value = splitInfo[wid*2 + 1];

	unsigned long processed = 0;
	int idx;
	int abs_index;
	int offset;
	const int MAX_THREADS_HALF = lsi / 2;
	const unsigned long ABS_NUM_VALUES = NUM_DATASETS * DS_LEN;
	int neededThreads;
	int part1_size;
	int part2_size;
	char ind;
	char category;

	if(lid == 0 || lid == 1) {
		partition_sizes[lid] = 0;
		cur_uncertainty[lid] = CALC_FACTOR;
	}

	neededThreads = min(NUM_CATEGORIES, MAX_THREADS_HALF);

	if(lid < neededThreads) {
		idx = lid;
		while(idx < NUM_CATEGORIES) {
			impurity_buffer_part1[idx] = 0;
			idx += neededThreads;
		}
	}
	else if(lid < neededThreads * 2) {
		idx = lid - neededThreads;
		while(idx < NUM_CATEGORIES) {
			impurity_buffer_part2[idx] = 0;
			idx += neededThreads;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// take chunks of size of MAX_LOCAL_MEMORY as long until whole NUM_DATASETS was processed
	while(processed < NUM_DATASETS) {
		idx = lid;
		while(idx < MAX_LOCAL_MEMORY) {
			abs_index = (processed + idx) * DS_LEN;
			if(abs_index + CATEGORY >= ABS_NUM_VALUES) {
				partition[idx] = OTHER;
				flags[idx] = 0; //INVALID_FLAG;
				idx += lsi;
				continue;
			}

			if(dataset[abs_index + feat] < value) {
				partition[idx] = PARTITION_ONE | dataset[abs_index + CATEGORY];
				flags[idx] = 1;
			} else {
				partition[idx] = PARTITION_TWO | dataset[abs_index + CATEGORY];
				flags[idx] = 0;
			}

			idx += lsi;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		offset = 1;
		while(offset <= MAX_LOCAL_MEMORY / 2) {
			idx = lid;

			while(idx < MAX_LOCAL_MEMORY - offset) {
				if(idx % (offset * 2) == 0) {
					flags[idx] += flags[idx+offset];
				}
				idx += lsi;
			}

			offset *= 2;
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		part1_size = flags[0];
		part2_size = min((ulong) MAX_LOCAL_MEMORY, (ulong) (NUM_DATASETS - processed)) - part1_size;

		if(lid == 0) {
			partition_sizes[0] += part1_size;
			partition_sizes[1] += part2_size;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		idx = lid;
		while(idx < MAX_LOCAL_MEMORY) {
			category = partition[idx];
			if((category & CATEGORY_PART) == OTHER) {
				idx += lsi;
				continue;
			}

			if((category & PARTITION_ONE) == PARTITION_ONE) {
				ind = category & CATEGORY_PART;
				atom_inc(&impurity_buffer_part1[ind]);
			} 
			else {
				ind = category & CATEGORY_PART;
				atom_inc(&impurity_buffer_part2[ind]);
			}
			
			idx += lsi;
		}

		processed += MAX_LOCAL_MEMORY;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(partition_sizes[0] == 0 || partition_sizes[1] == 0) {
		if(lid == 0) {
			splitInfoGain[wid] = 0.0;
		}
		return;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	neededThreads = min(NUM_CATEGORIES, MAX_THREADS_HALF);

	short imp_value;
	unsigned long reduce;

	if(lid < neededThreads) {
		idx = lid;

		while(idx < NUM_CATEGORIES) {
			imp_value = impurity_buffer_part1[idx];
			reduce = (long) (pow((double) imp_value / (double) partition_sizes[0], 2) * CALC_FACTOR);
			atom_sub(&cur_uncertainty[0], reduce);
			idx += neededThreads;
		}
	}
	else if(lid < neededThreads * 2) {
		idx = lid - neededThreads;
		while(idx < NUM_CATEGORIES) {
			imp_value = impurity_buffer_part2[idx];
			unsigned long reduce = (long) (pow((double) imp_value / (double) partition_sizes[1], 2) * CALC_FACTOR);
			atom_sub(&cur_uncertainty[1], reduce);
			idx += neededThreads;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(lid == 0) {
		double p = (double) partition_sizes[0] / (double) (partition_sizes[0] + partition_sizes[1]);
		double curUncertaintyPart1 = (double) cur_uncertainty[0] / (double) CALC_FACTOR;
		double curUncertaintyPart2 = (double) cur_uncertainty[1] / (double) CALC_FACTOR; 

		double infoGain = ((double) tempVals[2] - p * curUncertaintyPart1 - (double) (1 - p) * curUncertaintyPart2); // tempVals[2] = currentUncertainty

		splitInfoGain[wid] = (float) infoGain;

		// think of a way to determine the highest info gain on the gpu
	}
}