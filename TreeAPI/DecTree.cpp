#include "DecTree.h"

using std::string;
using std::to_string;

int tree::Node::numOfNodes = 0;

tree::Node::Node(NodeType type) : type(type) {
	numNode = ++numOfNodes;
}

tree::DecisionNode::DecisionNode(tree::Decision decision) : Node(DECISION), dec(decision) { };

string tree::DecisionNode::toString() {
	int trueBranchNum = 0;
	if (true_branch != NULL)
		trueBranchNum = true_branch->getNum();
	int falseBranchNum = 0;
	if (false_branch != NULL)
		falseBranchNum = false_branch->getNum();
	return "D(" + to_string(numNode) + "," + to_string(dec.refVal) + "," + to_string(dec.feature) + "," + to_string(trueBranchNum) + "," + to_string(falseBranchNum) + ")";
}

tree::ResultNode::ResultNode(tree::Result result) : Node(RESULT) {
	this->result.push_back(result);
	numResults = 1;
}

tree::ResultNode::ResultNode(std::vector<Result> result) : Node(RESULT) {
	this->result = result;
	numResults = (int)result.size();
}

string tree::ResultNode::toString() {
	std::stringstream ss;
	ss << "R(" << to_string(numNode);
	for (Result r : result) {
		ss << ",C(" << r.outcome << "," << to_string(r.probability) << ")";
	}
	ss << ")";

	return ss.str();
}

tree::Decision::Decision() : Decision(0, 0) {};

tree::Decision::Decision(int ref, int feat) : refVal(ref), feature(feat) {};

bool tree::Decision::decide(Record trData) {
	return trData.feature[feature] <= refVal;
}

void tree::decisionNode(std::vector<tree::NodeRefs>& noderefs, string data)
{
	string tmp;
	std::stringstream values;
	int nodeNum = 0, refVal = 0, feature = 0, trueBranch = 0, falseBranch;
	tmp = data.substr(2, data.length() - 3);

	for (int i = 0; i < tmp.length(); i++) {
		char c = tmp.at(i);


		if (c == ',')
			c = ' ';

		values << c;
	}

	values >> nodeNum >> refVal >> feature >> trueBranch >> falseBranch;

	tree::NodeRefs ref;
	ref.trueBranch = trueBranch;
	ref.falseBranch = falseBranch;
	ref.nodeNum = nodeNum;
	ref.node = (tree::Node*) new tree::DecisionNode(tree::Decision(refVal, feature));

	noderefs.push_back(ref);
}

void tree::resultNode(std::vector<tree::NodeRefs>& noderefs, string data)
{
	string tmp;
	std::stringstream values;

	char c;
	for (int i = 1; i < data.length(); i++) {
		c = data.at(i);

		if (c == 'C' || c == '(' || c == ')')
			continue;

		if (c == ',')
			c = ' ';

		values << c;
	}

	string numNode;
	string outcome;
	string probability;
	std::vector<tree::Result> results;

	values >> numNode;
	while (values >> outcome >> probability) {
		tree::Result r;
		r.outcome = outcome;
		r.probability = std::stof(probability.c_str());
		results.push_back(r);
	}

	tree::NodeRefs ref;
	ref.trueBranch = 0;
	ref.falseBranch = 0;
	ref.nodeNum = stoi(numNode);
	ref.node = (tree::Node*) new tree::ResultNode(results);

	noderefs.push_back(ref);
}

void tree::buildTree(tree::Node*& tree, std::vector<tree::NodeRefs>& noderefs)
{
	tree::NodeRefs& ref = noderefs.at(0);

	tree = ref.node;
	ref.node = NULL;
	if (ref.trueBranch != 0)
		addNode(tree->true_branch, noderefs, ref.trueBranch);
	if (ref.falseBranch != 0)
		addNode(tree->false_branch, noderefs, ref.falseBranch);
}

void tree::addNode(tree::Node*& node, std::vector<tree::NodeRefs>& noderefs, int nodeNum)
{
	tree::NodeRefs& ref = noderefs.at(nodeNum - 1);
	if (ref.nodeNum != nodeNum) {
		int diff = nodeNum - ref.nodeNum;
		ref = noderefs.at(ref.nodeNum + diff);

		if (ref.nodeNum != nodeNum) {
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