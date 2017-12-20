
#include "DecTree.h"

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

tree::ResultNode::ResultNode(vector<Result> result) : Node(RESULT) {
	this->result = result;
	numResults = (int)result.size();
}

string tree::ResultNode::toString() {
	stringstream ss;
	ss << "R(" << to_string(numNode);
	for (Result r : result) {
		ss << ",C(" << r.outcome << "," << to_string(r.probability) << ")";
	}
	ss << ")";

	return ss.str();
}

tree::Decision::Decision() : Decision(0, 0) {};

tree::Decision::Decision(int ref, int feat) : refVal(ref), feature(feat) {};

bool tree::Decision::decide(Dataset trData) {
	return trData.feature[feature] >= refVal;
}