#define NUM_FEATURES 10
#define NUM_VALUES_PER_FEATURE 1001

__kernel void CalcUniqueVals(__global const unsigned short* dataset, __global unsigned char* unique_vals_buffer) {
	int gid = get_global_id(0);
	int gsi = get_global_size(0);

	short index = gid % 11;
	if(index != 10) {	//index 10 is category!
		short value = dataset[gid];
		unique_vals_buffer[index * NUM_VALUES_PER_FEATURE + value] = 1;
	}
}