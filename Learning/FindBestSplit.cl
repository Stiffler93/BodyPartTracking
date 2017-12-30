
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define MAX_LOCAL_MEMORY 32768
#define DS_LEN 11
#define NUM_CATEGORIES 6
#define CALC_FACTOR 10000000000
#define CATEGORY 10
#define OTHER 777

__kernel void FindBestSplit(__global const unsigned short* dataset, __global double* tempVals, __global const unsigned short* splitInfo, 
			__global double* splitInfoGain) {

	int gid = get_global_id(0);
	int wid = get_group_id(0);
	int lid = get_local_id(0);
	int lsi = get_local_size(0);

	// uses too much memory!!!
	local unsigned short part1[NUM_DATASETS];
	local unsigned short part2[NUM_DATASETS];
	local unsigned short part1_flags[NEXT_SQUARE_OF_TWO];
	local unsigned short part2_flags[NEXT_SQUARE_OF_TWO];

	unsigned short feat = splitInfo[wid*2];
	unsigned short value = splitInfo[wid*2 + 1];

	int idx = lid;
	while(idx < NUM_DATASETS) {
		if(dataset[idx * DS_LEN + feat] < value) {
			part1[idx] = dataset[idx * DS_LEN + CATEGORY];
			part2[idx] = (unsigned short) OTHER;
			part1_flags[idx] = 1;
			part2_flags[idx] = 0;
		} else {
			part1[idx] = (unsigned short) OTHER;
			part2[idx] = dataset[idx * DS_LEN + CATEGORY];
			part1_flags[idx] = 0;
			part2_flags[idx] = 1;
		}

		idx += lsi;
	}

	while(idx < NEXT_SQUARE_OF_TWO) {
		part1_flags[idx] = 0;
		part2_flags[idx] = 0;
		idx += lsi;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// calculate information gain
	int offset = 1;

	const int MAX_THREADS_HALF = lsi / 2;
	int neededThreads = min(NEXT_SQUARE_OF_TWO / 2, MAX_THREADS_HALF);

	while(offset <= NEXT_SQUARE_OF_TWO / 2) {
		if(lid < neededThreads) {
			idx = lid;

			while(idx < NEXT_SQUARE_OF_TWO - offset) {
				if(idx % offset == 0) {
					part1_flags[idx] += part1_flags[idx+offset];
				}
				idx += neededThreads;
			}
		} 
		else if(lid < 2 * neededThreads) {
			idx = lid - neededThreads;

			while(idx < NEXT_SQUARE_OF_TWO - offset) {
				if(idx % offset == 0) {
					part2_flags[idx] += part2_flags[idx+offset];
				} 
				idx += neededThreads;
			}
		}

		offset *= 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int part1_size = part1_flags[0];
	int part2_size = part2_flags[0];

	if(part1_size == 0 || part2_size == 0) {
		if(lid == 0) {
			splitInfoGain[wid] = 0.0;
		}
		return;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	local unsigned int impurity_buffer_part1[NUM_CATEGORIES];
	local unsigned int impurity_buffer_part2[NUM_CATEGORIES];

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

	neededThreads = min(NUM_DATASETS, MAX_THREADS_HALF);
	
	if(lid < neededThreads) {
		idx = lid;
		while(idx < NUM_DATASETS) {
			short category = part1[idx];
			if(category == OTHER) {
				idx += neededThreads;
				continue;
			}

			atomic_inc(&impurity_buffer_part1[category]);
			idx += neededThreads;
		}
	}
	else if(lid < neededThreads * 2) {
		idx = lid - neededThreads;
		while(idx < NUM_DATASETS) {
			short category = part2[idx];
			if(category == OTHER) {
				idx += neededThreads;
				continue;
			}

			atomic_inc(&impurity_buffer_part2[category]);
			idx += neededThreads;
		}
	}

	local unsigned long cur_uncertainty[2]; // work with factor billion
	if(lid == 0 || lid == 1) {
		cur_uncertainty[lid] = CALC_FACTOR;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	neededThreads = min(NUM_CATEGORIES, MAX_THREADS_HALF);

	if(lid < neededThreads) {
		idx = lid;
		while(idx < NUM_CATEGORIES) {
			short value = impurity_buffer_part1[idx];
			unsigned long reduce = (long) (pow((double) value / (double) part1_size, (double) 2) * CALC_FACTOR);
			atom_sub(&cur_uncertainty[0], reduce);
			idx += neededThreads;
		}
	}
	else if(lid < neededThreads * 2) {
		idx = lid - neededThreads;
		while(idx < NUM_CATEGORIES) {
			short value = impurity_buffer_part2[idx];
			unsigned long reduce = (long) (pow((double) value / (double) part2_size, (double) 2) * CALC_FACTOR);
			atom_sub(&cur_uncertainty[1], reduce);
			idx += neededThreads;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(lid == 0) {
		double p = (double) part1_size / (double) (part1_size + part2_size);
		double curUncertaintyPart1 = (double) cur_uncertainty[0] / (double) CALC_FACTOR;
		double curUncertaintyPart2 = (double) cur_uncertainty[1] / (double) CALC_FACTOR; 

		double infoGain = ((double) tempVals[2] - p * curUncertaintyPart1 - (double) (1 - p) * curUncertaintyPart2); // tempVals[2] = currentUncertainty

		splitInfoGain[wid] = infoGain;

		// think of a way to determine the highest info gain on the gpu
	}
}