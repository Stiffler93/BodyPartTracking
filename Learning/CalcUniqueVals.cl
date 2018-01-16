#define NUM_VALUES_PER_FEATURE 1001

__kernel void CalcUniqueVals(__global const unsigned short* dataset, __global unsigned char* unique_vals_buffer, const unsigned int NUM_DATASETS) {
	//int gid = get_global_id(0);
	//int gsi = get_global_size(0);
	int lid = get_local_id(0);
	int lsi = get_local_size(0);
	int wid = get_group_id(0);
	int nug = get_num_groups(0);

	//local unsigned int counter[1];
	//if(lid == 0)
	//	counter[0] = 0;

	//barrier(CLK_LOCAL_MEM_FENCE);

	unsigned int nextDataset = wid * lsi + lid;
	
	while(nextDataset < NUM_DATASETS) {
		for(int f = 0; f < NUM_FEATURES; f++) {
			short value = dataset[nextDataset * DS_LEN + f];
			unique_vals_buffer[f * NUM_VALUES_PER_FEATURE + value] = 1;
		}
	
		nextDataset += (nug * lsi);
	}
	
	//atomic_inc(&counter[0]);
	//
	//barrier(CLK_LOCAL_MEM_FENCE);
	//
	//if(lid == 0)
	//	printf("Group %d, Num-Groups %d, Counter = %d\n", wid, nug, counter[0]);

	//short index = gid % DS_LEN;
	//if(index != CATEGORY) {
	//	short value = dataset[gid];
	//	unique_vals_buffer[index * NUM_VALUES_PER_FEATURE + value] = 1;
	//}
}