#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include "DecTree.h"
#include "CPUTrainingInterface.h"

using namespace tree;

void startParallelTraining(Dataset* trData, const unsigned int numTrData, Node*& rootNode);