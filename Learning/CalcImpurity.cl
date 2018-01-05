#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define DS_LEN 11
#define NUM_CATEGORIES 6
#define CALC_FACTOR 10000000000

__kernel void CalcImpurity(__global const unsigned short* dataset, __global unsigned int* impurity_buffer, const unsigned int NUM_DATASETS) {
	unsigned int gid = get_global_id(0);
	int lid = get_local_id(0);
	int lsi = get_local_size(0);
	int wid = get_group_id(0);

	local unsigned int loc_impurity_buffer[NUM_CATEGORIES];  

	short idx;

	idx = lid;
	while(idx < NUM_CATEGORIES) {
		loc_impurity_buffer[idx] = 0;
		idx += lsi;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(gid < NUM_DATASETS) {
		short category = dataset[gid * DS_LEN + DS_LEN - 1];
		atomic_inc(&loc_impurity_buffer[category]);
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	idx = lid;
	while(idx < NUM_CATEGORIES) {
		if(loc_impurity_buffer[idx] > 0) {
			atomic_add(&impurity_buffer[idx], loc_impurity_buffer[idx]);
		}
		idx += lsi;
	}
}


