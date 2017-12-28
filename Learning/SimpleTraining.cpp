#include "SimpleTraining.h"
#include <map>
#include <cmath>
#include "CategoryUtils.h"

float impurity(Dataset* trData, int numTrData) {
	map<string, int> category_counts;

	for (int i = 0; i < numTrData; i++) {
		string name = trData[i].outcome;
		
		std::map<string, int>::iterator it = category_counts.find(name);
		if (it == category_counts.end()) {
			printf("Insert %s with value 1.\n", name.c_str());
			category_counts.insert(std::pair<string, int>(name, 1));
		}
		else {
			it->second++;
			printf("Increase %s to value %d.\n", name.c_str(), it->second);
		}
	}

	double impurity = 1;
	double reduction;
	for (std::map<string, int>::iterator it = category_counts.begin(); it != category_counts.end(); ++it) {
		reduction = pow(((double) it->second / (double) numTrData), 2);
		printf("Reduce Impurity %lf ", impurity);
		impurity -= reduction;
		printf("by %lf to %lf.\n", reduction, impurity);
	}

	printf("Return Impurity >%lf<\n", impurity);

	return (float) impurity;
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

	printf("Partition split: %d/%d\n", partition.false_branch_size, partition.true_branch_size);

	float p = fBs / (fBs + tBs);
	printf("Impurity False Branch: \n");
	float impFalse = impurity(partition.false_branch, partition.false_branch_size);
	printf("Impurity True Branch: \n");
	float impTrue = impurity(partition.true_branch, partition.true_branch_size);

	float infoGain = (current_uncertainty - p * impFalse - (1 - p) * impTrue);

	printf("InfoGain = (%f - %f * %f - (1 - %f) * %f) = %f\n", current_uncertainty, p, impFalse, p, impTrue, infoGain);

	return infoGain;
}

UniqueValues calcUniqueVals(Dataset* trData, int numTrData, int feature) {
	UniqueValues values;

	for (int i = 0; i < numTrData; i++) {
		int val = trData[i].feature[feature];

		vector<int>::iterator it = values.vals.begin();
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
	printf("current Uncertainty: \n");
	float current_uncertainty = impurity(trData, numTrData);
	BestSplit split;

	Partition part;
	part.true_branch = new Dataset[numTrData];
	part.false_branch = new Dataset[numTrData];

	for (int feat = 0; feat < numFeatures(); feat++) {
		UniqueValues unVals = calcUniqueVals(trData, numTrData, feat);

		for (vector<int>::iterator it = unVals.vals.begin(); it != unVals.vals.end(); ++it) {
			Decision dec(*it, feat);

			partition(&part, trData, numTrData, dec);

			if (part.false_branch_size == 0 || part.true_branch_size == 0)
				continue;

			if (feat != 0)
				break;

			float gain = infoGain(part, current_uncertainty);

			//printf("Feature = %d, Value = %d, Current Uncertainty = %f, Information Gain = %f\n", feat, *it, current_uncertainty, gain);

			if (gain >= split.gain) {
				split.gain = gain;
				split.decision = dec;
			}
		}
	}

	delete[] part.true_branch;
	delete[] part.false_branch;

	return split;
}

void buildTree(Node*& decNode, Dataset* trData, int numTrData) {
	BestSplit split = findBestSplit(trData, numTrData);

	if (split.gain == 0) {

		if (numTrData == 1 || impurity(trData, numTrData) == 0) {
			Result res;
			res.outcome = trData[0].outcome;
			res.probability = 1.0;
			decNode = (Node*) new ResultNode(res);
		}
		else {
			map<string, int> results;
			for (int i = 0; i < numTrData; i++) {
				string category = trData[i].outcome;
				map<string, int>::iterator val = results.lower_bound(category);

				if (val != results.end() && !(results.key_comp()(category, val->first))) {
					val->second++;
				}
				else {
					results.insert(val, map<string, int>::value_type(category, 1));
				}
			}

			int size = (int)results.size();
			int sum = 0;
			for (auto it : results) {
				sum += it.second;
			}

			vector<Result> endRes;
			for (auto it : results) {
				Result r;
				r.outcome = it.first;
				r.probability = (float)it.second / (float)sum;
				endRes.push_back(r);
			}

			decNode = (Node*) new ResultNode(endRes);
		}

		delete[] trData;

		return;
	}

	decNode = (Node*) new DecisionNode(split.decision);

	Partition part;
	part.true_branch = new Dataset[numTrData];
	part.false_branch = new Dataset[numTrData];

	partition(&part, trData, numTrData, split.decision);

	delete[] trData;

	return;

	if (part.true_branch_size > 0)
		buildTree(decNode->true_branch, part.true_branch, part.true_branch_size);

	if (part.false_branch_size > 0)
		buildTree(decNode->false_branch, part.false_branch, part.false_branch_size);
}

void startSimpleTraining(Dataset* trData, int numTrData, Node*& rootNode)
{
	buildTree(rootNode, trData, numTrData);
}