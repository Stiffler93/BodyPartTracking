#include "Tests.h"
#include "CategoryUtils.h"
#include "TreeUtils.h"
#include <fstream>

using namespace tree;
using std::string;

void testWithTrainingData(tree::Node * decisionTree)
{
	printf("Start Test with own Training Data: \n");
	string dataset = datasetFile();
	std::ifstream features(dataset);

	int feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10,
		feat11, feat12, feat13, feat14, feat15, feat16, feat17, feat18, feat19, feat20,
		feat21, feat22, feat23, feat24, feat25, feat26;
	int category;
	int total = 0, correctClass = 0, falseClass = 0;
	std::vector<Result> results;

	while (features >> feat1 >> feat2 >> feat3 >> feat4 >> feat5 >> feat6 >> feat7 >> feat8 >> feat9 >> feat10 
			>> feat11 >> feat12 >> feat13 >> feat14 >> feat15 >> feat16 >> feat17 >> feat18 >> feat19 >> feat20
			>> feat21 >> feat22 >> feat23 >> feat24 >> feat25 >> feat26 >> category) {
		Record set;
		set.feature[0] = feat1;
		set.feature[1] = feat2;
		set.feature[2] = feat3;
		set.feature[3] = feat4;
		set.feature[4] = feat5;
		set.feature[5] = feat6;
		set.feature[6] = feat7;
		set.feature[7] = feat8;
		set.feature[8] = feat9;
		set.feature[9] = feat10;
		set.feature[10] = feat11;
		set.feature[11] = feat12;
		set.feature[12] = feat13;
		set.feature[13] = feat14;
		set.feature[14] = feat15;
		set.feature[15] = feat16;
		set.feature[16] = feat17;
		set.feature[17] = feat18;
		set.feature[18] = feat19;
		set.feature[19] = feat20;
		set.feature[20] = feat21;
		set.feature[21] = feat22;
		set.feature[22] = feat23;
		set.feature[23] = feat24;
		set.feature[24] = feat25;
		set.feature[25] = feat26;
		set.outcome = categoryOfValue(category);

		total++;

		results.clear();
		findResult(decisionTree, set, results);

		if (results.size() == 0) {
			falseClass++;
			//printf("No Result returned\n");
		}
		else {
			if (results.size() == 1) {
				Result res = results.at(0);
				if (res.outcome == set.outcome) {
					correctClass++;
				}
				else {
					//printf("Wrong result.\n");
					falseClass++;
				}
			}
			else {
				bool found = false;
				float prob = 0;

				for (Result res : results)
					if (res.outcome == set.outcome) {
						found = true;
						prob = res.probability;
					}

				if (found) {
					correctClass++;
					//printf("Found, but low probability: >%f<\n", prob);
				}
				else {
					falseClass++;
				}
			}
		}
	}

	std::printf("\n\nTest finished.\n");
	std::printf("Tested a total of %d Datasets.\n", total);
	std::printf("Correct: >%d<, False: >%d<\n", correctClass, falseClass);
	std::printf("Correctly classified %lf percent.\n", (float)correctClass / (float)total * 100);
}

void testWithTestData(tree::Node * decisionTree, tree::Record * testData, int numTestData)
{
	int correctClass = 0, falseClass = 0;

	for (int i = 0; i < numTestData; i++) {
		Record set = testData[i];
		std::vector<Result> results;
		findResult(decisionTree, set, results);

		bool found = false;
		for (Result r : results) {
			if (r.outcome == set.outcome) {
				found = true;
				break;
			}
		}

		if (found) {
			correctClass++;
		}
		else {
			falseClass++;
		}
	}

	std::printf("\n\nTest finished.\n");
	std::printf("Tested a total of %d Datasets.\n", numTestData);
	std::printf("Correct: >%d<, False: >%d<\n", correctClass, falseClass);
	std::printf("Correctly classified %lf percent.\n", (float)correctClass / (float)numTestData * 100);
}
