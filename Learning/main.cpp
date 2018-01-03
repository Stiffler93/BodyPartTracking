#include "DecTree.h"
#include "TreeSettings.h"
#include "TreeUtils.h"
#include <ctime>
#include "Tests.h"
#include "SimpleTraining.h"
#include "ParallelTraining.h"
#include "CategoryUtils.h"

using namespace tree;

bool checkTreeFile(string outputfile);
bool checkTrainingInput(string inputfile);
void readTrainingData(string inputfile, Dataset* trData, int* numTrData);

void test(Dataset test, Node* decisionTree);


int main_par(int argc, char** argv) {
	// test main for ParallelTraining
	Node* n;
	startParallelTraining(NULL, 0, n);

	return 0;
}

int main(int argc, char** argv) 
{
	printf("Start.\n");
	trace("Read training Data from folder " + datasetFile());

	if (!checkTreeFile(treeFile())) {
		printf("Exit Training Program.\n");
		return 2;
	}
	
	if (!checkTrainingInput(datasetFile())) {
		printf("Exit Training Program.\n");
		return 3;
	}

	const int numDS = numDatasets();
	Dataset *trData = new Dataset[numDS];
	int numTrData = 0;

	readTrainingData(datasetFile(), trData, &numTrData);

	if (isTraceActive()) {
		for (int i = 0; i < numTrData; i++) {
			trace(to_string(i + 1) + " >" + trData[i].toString() + "<");
		}
	}
	printf("Successfully parsed %d rows of data.\n", numTrData);

	printf("Start Training? (y/n)\n");
	string a; 
	cin >> a;

	if (a != "y" && a != "Y")
		return 0;

	clock_t begin = clock();

	Node* decisionTree = NULL;
	//startSimpleTraining(trData, numTrData, decisionTree);
	startParallelTraining(trData, numTrData, decisionTree);

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	printf("Training took %lf seconds!\n", elapsed_secs);

	string b; 
	cin >> b;

	testWithTrainingData(decisionTree);

	trace("Write training results to " + treeFile());
	//printTree(decisionTree);

	ofstream tree_file(treeFile());
	saveTree(decisionTree, tree_file);
	tree_file.close();

	freeTree(decisionTree);

	//trData must not be deleted, because is already during Training

	printf("SUCCESSFULLY built DECISION TREE\n");
	string c;
	cin >> c;

	return 0;
}

bool checkTreeFile(string outputfile)
{
	trace("checkTreeFile()");
	trace("Tree file = " + outputfile);

	struct stat buf;
	if (stat(outputfile.c_str(), &buf) == 0)
	{
		printf("%s already exists. You have to delete the file before training process can start.\n", outputfile.c_str());
		printf("Do you want to delete it? (y/n): ");

		string c;
		cin >> c;

		if (c == "y" || c == "Y") {
			if (remove(outputfile.c_str()) == 0) {
				printf("%s was successfully deleted.\n", outputfile.c_str());
				return true;
			}
		}

		printf("%s was not deleted.\n", outputfile.c_str());
		return false;
	}

	return true;
}

bool checkTrainingInput(string inputfile) 
{
	trace("checkTrainingInput()");

	struct stat buf;
	if (stat(inputfile.c_str(), &buf) != 0)
	{
		printf("%s does not exist!\n", inputfile.c_str());
		printf("Training data was not found!\n");
		return false;
	}

	return true;
}

void readTrainingData(string inputfile, Dataset* trData, int* numTrData)
{
	trace("readTrainingData()");

	int feat;
	int category;

	ifstream trDataset(inputfile);
	while (trDataset >> feat) {
		trData[*numTrData].feature[0] = feat;
		for (int i = 1; i < numFeatures(); i++) {
			trDataset >> feat;
			trData[*numTrData].feature[i] = feat;
		}
		trDataset >> category;
		trData[*numTrData].outcome = categoryOfValue(category);
		
		(*numTrData)++;
	}

	trace("Parsed input training data.");
}

void test(Dataset test, Node * decisionTree)
{
	trace("Test for " + test.outcome + " with values: >" + to_string(test.feature[0]) + "," + to_string(test.feature[1]) + "," + to_string(test.feature[2]) + "<");

	/*Node* node = decisionTree;
	while (!node->isResult()) {
		if (((DecisionNode*)node)->dec.decide(test)) {
			node = node->true_branch;
		}
		else {
			node = node->false_branch;
		}
	}*/

	vector<Result> results;
	findResult(decisionTree, test, results);

	if (results.size() == 1) {
		trace("Result is " + results[0].outcome);
		if (results[0].outcome == test.outcome) {
			trace("TEST was SUCCESSFUL");
		} else {
			trace("TEST FAILED");
		}
	}
	else {
		for (Result r : results) {
			trace("Result is " + r.outcome + " with a probability of " + to_string(r.probability));
		}
	}
}

