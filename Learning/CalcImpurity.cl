#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define DS_LEN 11
#define NUM_CATEGORIES 6
#define CALC_FACTOR 10000000000

__kernel void CalcImpurity(__global const unsigned short* dataset, __global unsigned int* impurity_buffer, __global double* tempVals) {
	int gid = get_global_id(0);
	int gsi = get_global_size(0);
	int lid = get_local_id(0);
	int lsi = get_local_size(0);
	int wid = get_group_id(0);
	int nug = get_num_groups(0);

	local unsigned int loc_impurity_buffer[NUM_CATEGORIES];  

	// optimize code when using larger localsizes
	// in current test local size is 3 and NUM_CATEGORIES is 6, so initialization can be static!
	loc_impurity_buffer[lid] = 0;
	loc_impurity_buffer[lid+3] = 0;	

	barrier(CLK_LOCAL_MEM_FENCE);

	short category = dataset[gid * DS_LEN + DS_LEN - 1];
	atomic_inc(&loc_impurity_buffer[category]);

	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_add(&impurity_buffer[lid], loc_impurity_buffer[lid]);
	atomic_add(&impurity_buffer[lid+3], loc_impurity_buffer[lid+3]);

	if(wid == nug - 1) { // last work group adds values last and then calculate current uncertainty
		barrier(CLK_LOCAL_MEM_FENCE);

		local unsigned long cur_uncertainty[1]; // work with factor billion
		if(lid == 0) {
			cur_uncertainty[0] = CALC_FACTOR;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		short value = impurity_buffer[lid];
		unsigned long reduce = (long) (pow((double) value / (double) gsi, (double) 2) * CALC_FACTOR);
		atom_sub(&cur_uncertainty[0], reduce);

		value = impurity_buffer[lid+3];
		reduce = (long) (pow((double) value / (double) gsi, (double) 2) * CALC_FACTOR);
		atom_sub(&cur_uncertainty[0], reduce);

		barrier(CLK_LOCAL_MEM_FENCE);

		if(lid == lsi - 1) {
			tempVals[2] = (double) cur_uncertainty[0] / (double) CALC_FACTOR;
		}
	}
}


