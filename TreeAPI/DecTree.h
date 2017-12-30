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
#include "CategoryUtils.h"

using namespace std;

namespace tree {

typedef struct Dataset {
	short feature[BPT_NUM_FEATURES]; // think about a better way to use numFeatures()
	string outcome;

	unsigned short* toArray() {
		const int number = BPT_NUM_FEATURES + 1;
		unsigned short arr[number];
		memcpy(arr, feature, sizeof(short) * BPT_NUM_FEATURES);
		arr[number - 1] = categoryToValue(outcome);

		return arr;
	};
	string toString() {
		stringstream ss;
		for (int i = 0; i < numFeatures(); i++)
			ss << feature[i] << " ";
		ss << categoryToValue(outcome);
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

#pragma once
class Decision {
public:
	Decision();
	Decision(int ref, int feat);
	bool decide(Dataset trData);
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

#pragma once
class Node {
public:
	Node(NodeType type);
	bool isResult() { return type == RESULT; }
	Node* false_branch, *true_branch;
	virtual string toString() = 0;
	int getNum() { return numNode; }
protected:
	NodeType type;
	static int numOfNodes;
	int numNode;
};

#pragma once
class DecisionNode : tree::Node {
public:
	DecisionNode(Decision decision);
	Decision dec;
	string toString();
};

#pragma once
class ResultNode : tree::Node {
public:
	ResultNode(Result result);
	ResultNode(vector<Result> result);
	string toString();
	vector<Result> result;
	int numResults;
};

}


#endif