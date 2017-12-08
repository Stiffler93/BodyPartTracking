#ifndef DEC_TREE
#define DEC_TREE

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <functional>
#include <vector>
#include <map>
#include "TreeSettings.h"

using namespace std;



namespace tree {

typedef struct Dataset {
	int feature[BPT_NUM_FEATURES]; // think about a better way to use numFeatures()
	string outcome;
	string toString() {
		stringstream ss;
		for (int i = 0; i < numFeatures(); i++)
			ss << feature[i] << ",";
		ss << outcome;
		return ss.str();
	};
} Dataset;

typedef struct UniqueValues {
	vector<int> vals;
	int numVals = 0;
} UniqueValues;

typedef struct Result {
	string outcome = "";
	float probability = 0;
} Result;

typedef struct Partition {
	Dataset *true_branch, *false_branch;
	int true_branch_size = 0, false_branch_size = 0;
} Partition;

class Decision {
public:
	Decision() : Decision(0, 0) {};
	Decision(int ref, int feat) : refVal(ref), feature(feat) {};
	bool decide(Dataset trData) {
		return trData.feature[feature] >= refVal;
	}
	int refVal, feature;
};

typedef struct BestSplit {
public:
	float gain = 0;
	Decision decision;
} BestSplit;

enum NodeType {
	DECISION = 0,
	RESULT = 1
};

class Node {
public:
	Node(NodeType type) : type(type) { numNode = ++numOfNodes; };
	bool isResult() { return type == RESULT; }
	Node* false_branch, *true_branch;
	virtual string toString() = 0;
	int getNum() { return numNode; }
protected:
	NodeType type;
	static int numOfNodes;
	int numNode;
};

int tree::Node::numOfNodes = 0;

class DecisionNode : tree::Node {
public:
	DecisionNode(Decision decision) : Node(DECISION), dec(decision) { }
	Decision dec;
	string toString() {
		int trueBranchNum = true_branch->getNum();
		int falseBranchNum = false_branch->getNum();
		return "D(" + to_string(numNode) + "," + to_string(dec.refVal) + "," + to_string(dec.feature) + "," + to_string(trueBranchNum) + "," + to_string(falseBranchNum) + ")";
	}
};

class ResultNode : tree::Node {
public:
	ResultNode(Result result) : Node(RESULT) {
		this->result.push_back(result);
		numResults = 1;
	}
	ResultNode(vector<Result> result) : Node(RESULT) {
		this->result = result;
		numResults = (int)result.size();
	}
	string toString() {
		stringstream ss;
		ss << "R(" << to_string(numNode);
		for (Result r : result) {
			ss << ",C(" << r.outcome << "," << to_string(r.probability) << ")";
		}
		ss << ")";

		return ss.str();
	}
	vector<Result> result;
	int numResults;
};

}


#endif