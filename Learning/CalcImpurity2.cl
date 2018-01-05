#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define DS_LEN 11
#define NUM_CATEGORIES 6
#define CALC_FACTOR 10000000000

__kernel void CalcImpurity2(__global unsigned int* impurity_buffer, __global double* tempVals, const unsigned int NUM_DATASETS) {

	int lid = get_local_id(0);
	int lsi = get_local_size(0);
	
	local unsigned long cur_uncertainty[1]; // work with factor billion
	if(lid == 0) {
		cur_uncertainty[0] = CALC_FACTOR;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	short idx = lid;
	while(idx < NUM_CATEGORIES) {
		unsigned int value = impurity_buffer[idx];
		unsigned long reduce = (long) (pow((double) value / (double) NUM_DATASETS, (double) 2) * CALC_FACTOR);
		atom_sub(&cur_uncertainty[0], reduce);
		idx += lsi;
		//printf("Reduce uncertainty IDX = %d> Value = %d, reduce = %ld\n", idx, value, reduce);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(lid == 0) {
		tempVals[2] = (double) cur_uncertainty[0] / (double) CALC_FACTOR;
	}
}