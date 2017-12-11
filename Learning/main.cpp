#include "DecTree.h"
#include "TreeSettings.h"
#include "TreeUtils.h"
#include <ctime>

using namespace tree;

ofstream traceFile(debugFile());

void trace(string trace);
void trace(const char* trace);

bool checkTreeFile(string outputfile);
bool checkTrainingInput(string inputfile);
void readTrainingData(string inputfile, Dataset* trData, int* numTrData);
void startTraining(Dataset* trData, int numTrData, Node*& rootNode);
void saveTree(Node* tree, ofstream& file);

void test(Dataset test, Node* decisionTree);


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
	startTraining(trData, numTrData, decisionTree);

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	printf("Training took %lf seconds!\n", elapsed_secs);

	trace("Write training results to " + treeFile());
	printTree(decisionTree);

	ofstream tree_file(treeFile());
	saveTree(decisionTree, tree_file);
	tree_file.close();

	traceFile.close();

	//trData must not be deleted, because is already during Training

	printf("SUCCESSFULLY built DECISION TREE\n");
	printf("Press enter...");
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
	string category;

	ifstream trDataset(inputfile);
	while (trDataset >> feat) {
		trData[*numTrData].feature[0] = feat;
		for (int i = 1; i < numFeatures(); i++) {
			trDataset >> feat;
			trData[*numTrData].feature[i] = feat;
		}
		trDataset >> category;
		trData[*numTrData].outcome = category;
		
		(*numTrData)++;
	}

	trace("Parsed input training data.");
}

float impurity(Dataset* trData, int numTrData) {
	trace("impuritiy(); numTrData = " + to_string(numTrData));

	vector<string> classes;

	for (int i = 0; i < numTrData; i++) {
		string name = trData[i].outcome;

		vector<string>::iterator it = classes.begin();
		for (int index = 0; it != classes.end(); ++it, ++index) {
			if (*it == name)
				break;

			if (*it > name) {
				classes.insert(it, name);
				break;
			}
		}
		
		if (it == classes.end()) {
			classes.push_back(name);
		}
	}

	float numClasses = (float) classes.size();

	return (float) 1 - (1 / numClasses);
}

void partition(Partition* part, Dataset* trData, int numTrData, Decision decision) {
	trace("partition(), numTrData = " + to_string(numTrData));

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

	trace("True_Branch_Size = " + to_string(part->true_branch_size) + ", False_Branch_Size = " + to_string(part->false_branch_size));
}

float infoGain(Partition partition, float current_uncertainty) {
	trace("infoGain(), current uncertainty = " + to_string(current_uncertainty));

	float fBs = (float) partition.false_branch_size;
	float tBs = (float) partition.true_branch_size;

	float p = fBs / (fBs + tBs);
	float impFalse = impurity(partition.false_branch, partition.false_branch_size);
	float impTrue = impurity(partition.true_branch, partition.true_branch_size);

	float infoGain = (current_uncertainty - p * impFalse - (1 - p) * impTrue);
	trace("infoGain == " + to_string(infoGain));

	return infoGain;
}

UniqueValues calcUniqueVals(Dataset* trData, int numTrData, int feature) {
	trace("calcUniqueVals()");

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

	values.numVals = (int) values.vals.size();

	trace("Found " + to_string(values.numVals) + " different Values");

	return values;
}

BestSplit findBestSplit(Dataset* trData, int numTrData) {
	trace("findBestSplit()");

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

			trace("findBestSplit(), after partition: True Branch size = " + to_string(part.true_branch_size) + ", False Branch size = " + to_string(part.false_branch_size));

			if (part.false_branch_size == 0 || part.true_branch_size == 0)
				continue;
				
			float gain = infoGain(part, current_uncertainty);

			if (gain >= split.gain) {
				trace("Gain(" + to_string(gain) + " >= split.gain(" + to_string(split.gain) + ")\n");
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
	trace("buildTree(); training Data size = " + to_string(numTrData));
	
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

			int size = (int) results.size();
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

	printf("Node found.\n");

	if(part.true_branch_size > 0)
		buildTree(decNode->true_branch, part.true_branch, part.true_branch_size);
	
	if(part.false_branch_size > 0)
		buildTree(decNode->false_branch, part.false_branch, part.false_branch_size);
}

void startTraining(Dataset* trData, int numTrData, Node*& rootNode)
{
	trace("startTraining()");
	buildTree(rootNode, trData, numTrData);
}

void saveTree(Node * tree, ofstream& file)
{
	file << tree->toString() << endl;

	if (tree->true_branch != NULL)
		saveTree(tree->true_branch, file);
	if (tree->false_branch != NULL)
		saveTree(tree->false_branch, file);

	delete tree;
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

void trace(string trace)
{
	if (isTraceActive())
		traceFile << trace << endl;
}

void trace(const char * trace)
{
	if (isTraceActive())
		traceFile << trace << endl;
}
