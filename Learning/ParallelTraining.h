#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include "DecTree.h"
#include "CPUTrainingInterface.h"

void startParallelTraining(tree::Record* trData, const unsigned int numTrData, tree::Node*& rootNode);