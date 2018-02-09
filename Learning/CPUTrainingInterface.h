#include "DecTree.h"

bool isPure(tree::Record* trData, int numTrData);

float impurity(tree::Record* trData, int numTrData);

void partition(tree::Partition* part, tree::Record* trData, int numTrData, tree::Decision decision);

float infoGain(tree::Partition partition, float current_uncertainty);

tree::UniqueValues calcUniqueVals(tree::Record* trData, int numTrData, int feature);

tree::BestSplit findBestSplit(tree::Record* trData, int numTrData);