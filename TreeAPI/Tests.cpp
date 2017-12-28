#include "Tests.h"
#include "CategoryUtils.h"

using namespace tree;

void testWithTrainingData(tree::Node * decisionTree)
{
	printf("Start Test with own Training Data: \n");
	string dataset = datasetFile();
	ifstream features(dataset);

	int feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10;
	int category;
	int total = 0, correctClass = 0, falseClass = 0;
	vector<Result> results;

	while (features >> feat1 >> feat2 >> feat3 >> feat4 >> feat5 >> feat6 >> feat7 >> feat8 >> feat9 >> feat10 >> category) {
		Dataset set;
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

	printf("\n\nTest finished.\n");
	printf("Tested a total of %d Datasets.\n", total);
	printf("Correct: >%d<, False: >%d<\n", correctClass, falseClass);
	printf("Correctly classified %lf percent.\n", (float)correctClass / (float)total * 100);
}