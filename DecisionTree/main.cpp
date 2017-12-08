#include "DecTree.h"
#include "TreeSettings.h"
#include <fstream>
#include <string>

using namespace tree;

typedef struct NodeRefs {
	Node* node;
	int trueBranch = 0, falseBranch = 0, nodeNum = 0;
}NodeRefs;

void decisionNode(vector<NodeRefs> *noderefs, string data);
void resultNode(vector<NodeRefs> *noderefs, string data);
void buildTree(Node* tree, vector<NodeRefs> *noderefs);

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

	Node* tree = NULL;
	buildTree(tree, &nodes);

	printf("Tree successfully reconstructed!\n");

	string s;
	cin >> s;

	return 0;
}

void decisionNode(vector<NodeRefs> *noderefs, string data)
{
	printf("Decision Node: \n");
	string tmp;
	stringstream values;
	int nodeNum = 0, refVal = 0, feature = 0, trueBranch = 0, falseBranch;
	tmp = data.substr(2, data.length() - 3);
	printf("Tmp: >%s<\n", tmp.c_str());

	
	for (int i = 0; i < tmp.length(); i++) {
		char c = tmp.at(i);
		

		if (c == ',')
			c = ' ';

		values << c;
	}

	values >> nodeNum >> refVal >> feature >> trueBranch >> falseBranch;

	//printf("refVal: %d, feature: %d, trueBanch: %d, falseBranch: %d\n", refVal, feature, trueBranch, falseBranch);

	NodeRefs ref;
	ref.trueBranch = trueBranch;
	ref.falseBranch = falseBranch;
	ref.nodeNum = nodeNum;

	ref.node = (Node*) new DecisionNode(Decision(refVal, feature));
	printf("DecisionNode: >%s<\n", ref.node->toString().c_str());

	noderefs->push_back(ref);
}

void resultNode(vector<NodeRefs> *noderefs, string data)
{
	printf("Result Node: \n");

	string tmp;
	stringstream values;
	tmp = data.substr(2, data.length() - 3);
	printf("Tmp: >%s<\n", tmp.c_str());

	char c;
	for (int i = 0; i < data.length(); i++) {
		c = data.at(i);

		if (c == 'C' || c == '(' || c == ')')
			continue;
		
		if (c == ',')
			c = ' ';

		values << c;
	}

	int numNode = 0;
	string outcome;
	float probability;
	vector<Result> results;

	values >> numNode;
	while (values >> outcome >> probability) {
		Result r;
		r.outcome = outcome;
		r.probability = probability;
		results.push_back(r);
	}
	
	NodeRefs ref;
	ref.trueBranch = 0;
	ref.falseBranch = 0;
	ref.nodeNum = numNode;
	ref.node = (Node*) new ResultNode(results);
	printf("ResultNode: >%s<\n", ref.node->toString());

	noderefs->push_back(ref);
}

void buildTree(Node * tree, vector<NodeRefs>* noderefs)
{
}
