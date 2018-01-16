#include "CPUTrainingInterface.h"
#include "TreeUtils.h"
#include <map>

using namespace tree;
using std::string;

bool isPure(tree::Dataset* trData, int numTrData) {
	string ref = trData[0].outcome;
	for (int i = 1; i < numTrData; i++) {
		string name = trData[i].outcome;
		if (ref != name)
			return false;
	}

	return true;
}

float impurity(Dataset* trData, int numTrData) {
	std::map<string, int> category_counts;

	for (int i = 0; i < numTrData; i++) {
		string name = trData[i].outcome;

		std::map<string, int>::iterator it = category_counts.find(name);
		if (it == category_counts.end()) {
			category_counts.insert(std::pair<string, int>(name, 1));
		}
		else {
			it->second++;
		}
	}

	double impurity = 1;
	double reduction;
	for (std::map<string, int>::iterator it = category_counts.begin(); it != category_counts.end(); ++it) {
		reduction = pow(((double)it->second / (double)numTrData), 2);
		impurity -= reduction;
	}

	return (float)impurity;
}

void partition(Partition* part, Dataset* trData, int numTrData, Decision decision) {
	part->false_branch_size = part->true_branch_size = 0;

	for (int i = 0; i < numTrData; i++) {
		if (decision.decide(trData[i])) {
			part->true_branch[part->true_branch_size] = trData[i];
			part->true_branch_size++;
		}
		else {
			part->false_branch[part->false_branch_size] = trData[i];
			part->false_branch_size++;
		}
	}
}

float infoGain(Partition partition, float current_uncertainty) {
	float fBs = (float)partition.false_branch_size;
	float tBs = (float)partition.true_branch_size;

	float p = fBs / (fBs + tBs);
	float impFalse = impurity(partition.false_branch, partition.false_branch_size);
	float impTrue = impurity(partition.true_branch, partition.true_branch_size);

	float infoGain = (current_uncertainty - p * impFalse - (1 - p) * impTrue);

	if (infoGain < -0.0001 || infoGain > 1) {
		printf("InfoGain wrong?!\n");
		std::printf("InfoGain = (%f - %f * %f - (1 - %f) * %f) = %f\n", current_uncertainty, p, impFalse, p, impTrue, infoGain);
	}
	return infoGain;
}

UniqueValues calcUniqueVals(Dataset* trData, int numTrData, int feature) {
	UniqueValues values;

	for (int i = 0; i < numTrData; i++) {
		int val = trData[i].feature[feature];

		std::vector<int>::iterator it = values.vals.begin();
		for (; it != values.vals.end(); ++it) {
			if (*it == val)
				break;

			if (*it > val) {
				values.vals.insert(it, val);
				break;
			}
		}

		if (it == values.vals.end()) {
			values.vals.push_back(val);
		}
	}

	values.numVals = (int)values.vals.size();

	return values;
}

BestSplit findBestSplit(Dataset* trData, int numTrData) {
	float current_uncertainty = impurity(trData, numTrData);
	BestSplit split;

	Partition part;
	part.true_branch = new Dataset[numTrData];
	part.false_branch = new Dataset[numTrData];

	int uniqueVal = 0;

	//trace("Unique Vals: ");

	for (int feat = 0; feat < numFeatures(); feat++) {
		UniqueValues unVals = calcUniqueVals(trData, numTrData, feat);

		//for (int i = 0; i < unVals.numVals; i++) {
		//	trace("UniqueVal: >" + std::to_string(feat) + "," + std::to_string(unVals.vals[i]) + "<");
		//}

		for (std::vector<int>::iterator it = unVals.vals.begin(); it != unVals.vals.end(); ++it) {
			uniqueVal++;

			Decision dec(*it, feat);

			partition(&part, trData, numTrData, dec);

			if (part.false_branch_size == 0 || part.true_branch_size == 0)
				continue;

			float gain = infoGain(part, current_uncertainty);
			const int FACTOR = 1000000;
			int newGain = (int)(gain * FACTOR);
			int oldGain = (int)(split.gain * FACTOR);

			if (/*gain > split.gain*/newGain > oldGain) {
				split.gain = gain;
				split.decision = dec;
			}
		}
	}

	delete[] part.true_branch;
	delete[] part.false_branch;

	return split;
}