#include "SimpleTraining.h"
#include "CategoryUtils.h"
#include "CPUTrainingInterface.h"
#include <map>
#include "TreeUtils.h"

using namespace tree;
using std::string;

void buildTree(Node*& decNode, Record* trData, int numTrData, unsigned long* numTrDataLeft) {

	float imp = impurity(trData, numTrData);

	BestSplit split;
	if(imp > BPT_STOP_EVALUATION_IMPURITY)
		split = findBestSplit(trData, numTrData);

	if (/*imp <= BPT_STOP_EVALUATION_IMPURITY || numTrData <= BPT_STOP_EVALUATION_LIMIT || */split.gain == 0) {
		if (numTrData == 1 || isPure(trData, numTrData)) {
			Result res;
			res.outcome = trData[0].outcome;
			res.probability = 1.0;
			decNode = (Node*) new ResultNode(res);
		}
		else {
			std::map<string, int> results;
			for (int i = 0; i < numTrData; i++) {
				string category = trData[i].outcome;
				std::map<string, int>::iterator val = results.lower_bound(category);

				if (val != results.end() && !(results.key_comp()(category, val->first))) {
					val->second++;
				}
				else {
					results.insert(val, std::map<string, int>::value_type(category, 1));
				}
			}

			int size = (int)results.size();
			int sum = 0;
			for (auto it : results) {
				sum += it.second;
			}

			std::vector<Result> endRes;
			for (auto it : results) {
				Result r;
				r.outcome = it.first;
				r.probability = (float)it.second / (float)sum;
				endRes.push_back(r);
			}

			decNode = (Node*) new ResultNode(endRes);
		}

		delete[] trData;

		*numTrDataLeft = *numTrDataLeft - numTrData;
		printf("NumTrDataLeft: %ld\n", *numTrDataLeft);

		return;
	}

	decNode = (Node*) new DecisionNode(split.decision);

	Partition part;
	part.true_branch = new Record[numTrData];
	part.false_branch = new Record[numTrData];

	partition(&part, trData, numTrData, split.decision);

	trace("Decision >" + std::to_string(split.decision.feature) + "|" + std::to_string(split.decision.refVal) + "<");
	trace("False_Branch_size = " + std::to_string(part.false_branch_size) + ", True_Branch_size = " + std::to_string(part.true_branch_size));

	delete[] trData;

	if (part.true_branch_size > 0)
		buildTree(decNode->true_branch, part.true_branch, part.true_branch_size, numTrDataLeft);

	if (part.false_branch_size > 0)
		buildTree(decNode->false_branch, part.false_branch, part.false_branch_size, numTrDataLeft);
}

void startSimpleTraining(Record* trData, int numTrData, Node*& rootNode)
{
	unsigned long numTrDataLeft = numTrData;
	printf("NumTrDataLeft: %ld\n", numTrDataLeft);

	buildTree(rootNode, trData, numTrData, &numTrDataLeft);
}