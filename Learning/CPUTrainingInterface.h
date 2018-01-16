#include "DecTree.h"

bool isPure(tree::Dataset* trData, int numTrData);

float impurity(tree::Dataset* trData, int numTrData);

void partition(tree::Partition* part, tree::Dataset* trData, int numTrData, tree::Decision decision);

float infoGain(tree::Partition partition, float current_uncertainty);

tree::UniqueValues calcUniqueVals(tree::Dataset* trData, int numTrData, int feature);

tree::BestSplit findBestSplit(tree::Dataset* trData, int numTrData);