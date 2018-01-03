#include "SimpleTraining.h"
#include <map>
#include <cmath>
#include "CategoryUtils.h"
#include <ctime>

void buildTree(Node*& decNode, Dataset* trData, int numTrData, unsigned long* numTrDataLeft) {
	//clock_t begin = clock();

	//std::printf("Call buildTree(). numTrData = %d\n", numTrData);
	//std::printf("TrData: \n");
	bool isHeterogenous = false;
	string temp = trData[0].outcome;
	for (int i = 1; i <= numTrData; i++) {
		if (temp != trData[i - 1].outcome) {
			isHeterogenous = true;
			break;
		}
		//std::printf("\t%d.: >%s<\n", i, trData[i - 1].toString().c_str());
	}

	BestSplit split;
	if(isHeterogenous)
		split = findBestSplit(trData, numTrData);

	if (numTrData <= BPT_STOP_EVALUATION_LIMIT || split.gain == 0) {
		if (numTrData == 1 || isPure(trData, numTrData)) {
			Result res;
			res.outcome = trData[0].outcome;
			res.probability = 1.0;
			//std::printf("--> ResultNode(%s,%lf).\n", res.outcome.c_str(), res.probability);
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

			//std::printf("--> ResultNode(%s,%lf).\n", endRes[0].outcome.c_str(), endRes[0].probability);
			decNode = (Node*) new ResultNode(endRes);
		}

		delete[] trData;

		*numTrDataLeft = *numTrDataLeft - numTrData;
		printf("NumTrDataLeft: %ld\n", *numTrDataLeft);

		return;
	}

	//std::printf("--> DecisionNode(%d,%d).\n", split.decision.feature, split.decision.refVal);
	decNode = (Node*) new DecisionNode(split.decision);

	Partition part;
	part.true_branch = new Dataset[numTrData];
	part.false_branch = new Dataset[numTrData];

	partition(&part, trData, numTrData, split.decision);

	//std::printf("True-Branch Split: \n");
	//for (int i = 0; i < part.true_branch_size; i++) {
	//	std::printf("\t%d.: >%s<\n", i, part.true_branch[i].toString().c_str());
	//}

	//std::printf("False-Branch Split: \n");
	//for (int i = 0; i < part.false_branch_size; i++) {
	//	std::printf("\t%d.: >%s<\n", i, part.false_branch[i].toString().c_str());
	//}

	//std::printf("\n");

	delete[] trData;

	//clock_t end = clock();
	//double elapsed_secs = double(end - begin) / (double)CLOCKS_PER_SEC;
	//printf("Cycle took %lf seconds!\n", elapsed_secs);

	if (part.true_branch_size > 0)
		buildTree(decNode->true_branch, part.true_branch, part.true_branch_size, numTrDataLeft);

	if (part.false_branch_size > 0)
		buildTree(decNode->false_branch, part.false_branch, part.false_branch_size, numTrDataLeft);
}

void startSimpleTraining(Dataset* trData, int numTrData, Node*& rootNode)
{
	unsigned long numTrDataLeft = BPT_NUM_DATASETS;
	printf("NumTrDataLeft: %ld\n", numTrDataLeft);

	buildTree(rootNode, trData, numTrData, &numTrDataLeft);
}