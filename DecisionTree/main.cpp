#include "DecTree.h"
#include "TreeSettings.h"
#include <fstream>
#include <string>
#include "TreeConstants.h"
#include "TreeUtils.h"
#include "opencv2\opencv.hpp"
#include "ImageRecorder.h"

using namespace tree;

typedef struct NodeRefs {
	tree::Node* node;
	int trueBranch = 0, falseBranch = 0, nodeNum = 0;
}NodeRefs;

void decisionNode(vector<NodeRefs> *noderefs, string data);
void resultNode(vector<NodeRefs> *noderefs, string data);
void buildTree(tree::Node*& tree, vector<NodeRefs> *noderefs);
void addNode(tree::Node*& node, vector<NodeRefs> *noderefs, int nodeNum);
void test(tree::Node* decisionTree);
void realTest(tree::Node* decisionTree);

int main(int argc, char** argv) 
{
	vector<NodeRefs> nodes;

	ifstream treeFile(treeFile());
	string data = "";
	while (treeFile >> data) {
		if (data.empty())
			break;

		printf("Data: >%s<\n", data.c_str());

		if (data.substr(0, 1) == "D") {
			decisionNode(&nodes, data);
		}
		else if (data.substr(0, 1) == "R") {
			resultNode(&nodes, data);
		}
		else {
			throw exception("Unrecognized Tree Node! Only DecisionNodes [D] and ResultNodes [R] are valid!");
		}
	}

	tree::Node* tree = NULL;
	buildTree(tree, &nodes);

	printf("Tree successfully reconstructed!\n");

	/*printf("Tree: \n");
	printTree(tree);*/

	printf("Start Decision Tree Test: \n");

	test(tree);

	string s;
	cin >> s;

	return 0;
}

void decisionNode(vector<NodeRefs> *noderefs, string data)
{
	string tmp;
	stringstream values;
	int nodeNum = 0, refVal = 0, feature = 0, trueBranch = 0, falseBranch;
	tmp = data.substr(2, data.length() - 3);

	for (int i = 0; i < tmp.length(); i++) {
		char c = tmp.at(i);
		

		if (c == ',')
			c = ' ';

		values << c;
	}

	values >> nodeNum >> refVal >> feature >> trueBranch >> falseBranch;

	NodeRefs ref;
	ref.trueBranch = trueBranch;
	ref.falseBranch = falseBranch;
	ref.nodeNum = nodeNum;
	ref.node = (tree::Node*) new DecisionNode(Decision(refVal, feature));

	noderefs->push_back(ref);
}

void resultNode(vector<NodeRefs> *noderefs, string data)
{
	string tmp;
	stringstream values;

	char c;
	for (int i = 0; i < data.length(); i++) {
		c = data.at(i);

		if (c == 'C' || c == '(' || c == ')' || c == 'R')
			continue;
		
		if (c == ',')
			c = ' ';

		values << c;
	}

	string numNode;
	string outcome;
	string probability;
	vector<Result> results;

	values >> numNode;
	while (values >> outcome >> probability) {
		Result r;
		r.outcome = outcome;
		r.probability = stof(probability.c_str());
		results.push_back(r);
	}
	
	NodeRefs ref;
	ref.trueBranch = 0;
	ref.falseBranch = 0;
	ref.nodeNum = stoi(numNode);
	ref.node = (tree::Node*) new ResultNode(results);

	noderefs->push_back(ref);
}

void buildTree(tree::Node*& tree, vector<NodeRefs>* noderefs)
{
	printf("build tree: \n");
	NodeRefs ref = noderefs->at(0);
	tree = ref.node;
	if (ref.trueBranch != 0)
		addNode(tree->true_branch, noderefs, ref.trueBranch);
	if (ref.falseBranch != 0)
		addNode(tree->false_branch, noderefs, ref.falseBranch);
}

void addNode(tree::Node*& node, vector<NodeRefs>* noderefs, int nodeNum)
{
	printf("Add Node with num %d\n", nodeNum - 1);
	NodeRefs ref = noderefs->at(nodeNum - 1);
	if (ref.nodeNum != nodeNum) {
		printf("Correct node not immediately found!\n");
		int diff = nodeNum - ref.nodeNum;
		ref = noderefs->at(ref.nodeNum + diff);

		if (ref.nodeNum != nodeNum) {
			printf("Node not found!\n");
			return;
		}
	}

	node = ref.node;
	ref.node = NULL;

	if (ref.trueBranch != 0)
		addNode(node->true_branch, noderefs, ref.trueBranch);
	if (ref.falseBranch != 0)
		addNode(node->false_branch, noderefs, ref.falseBranch);
}

void test(tree::Node * decisionTree)
{
	string dataset = datasetFile();
	ifstream features(dataset);

	int feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10;
	string category;
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
		printf("Test Dataset: >%s<\n", set.toString().c_str());

		results.clear();
		findResult(decisionTree, set, results);

		total++;

		if (results.size() == 0) {
			falseClass++;
			printf("No Result returned\n");
		}
		else {
			Result res = results.at(0);
			if (res.outcome == category) {
				printf("Correct result.\n");
				correctClass++;
			}
			else {
				printf("Wrong result.\n");
				falseClass++;
			}
		}
	}

	printf("\n\nTest finished.\n");
	printf("Tested a total of %d Datasets.\n", total);
	printf("Correct: >%d<, False: >%d<\n", correctClass, falseClass);
	printf("Correctly classified %lf percent.\n", (float)correctClass / (float)total * 100);
}

void realTest(tree::Node * decisionTree)
{
}
