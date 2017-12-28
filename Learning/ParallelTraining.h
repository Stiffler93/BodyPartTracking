#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include "DecTree.h"

using namespace tree;

void startParallelTraining(Dataset* trData, int numTrData, Node*& rootNode);