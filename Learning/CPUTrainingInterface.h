#include "DecTree.h"
#include <map>
#include <cmath>
#include "CategoryUtils.h"
#include <ctime>

using namespace tree;

bool isPure(Dataset* trData, int numTrData);

float impurity(Dataset* trData, int numTrData);

void partition(Partition* part, Dataset* trData, int numTrData, Decision decision);

float infoGain(Partition partition, float current_uncertainty);

UniqueValues calcUniqueVals(Dataset* trData, int numTrData, int feature);

BestSplit findBestSplit(Dataset* trData, int numTrData);