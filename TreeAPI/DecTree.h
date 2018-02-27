#ifndef DEC_TREE
#define DEC_TREE

#include "TreeSettings.h"
#include "CategoryUtils.h"
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>

namespace tree {

typedef struct MetaData {
	unsigned short type = 0;
	unsigned int numRecords = 0;
	void ofString(std::string s) {
		std::stringstream ss;
		ss << s;
		ss >> s; // remove '##'
		ss >> type;
		ss >> numRecords;
	}
	std::string toString() {
		std::stringstream ss;
		ss << "## " << type << " " << numRecords;
		return ss.str();
	}
} MetaData;

typedef struct DatasetMetaData {
	MetaData meta[NUM_META_DATA];
	std::string toString() {
		std::stringstream ss;
		for (int i = 0; i < NUM_META_DATA; i++) {
			ss << meta[i].toString() << std::endl;
		}
		return ss.str();
	}
} DatasetMetaData;

typedef struct Record {
	short feature[BPT_NUM_FEATURES]; // think about a better way to use numFeatures()
	std::string outcome;

	unsigned short* toArray() {
		const int number = BPT_NUM_FEATURES + 1;
		unsigned short arr[number];
		memcpy(arr, feature, sizeof(short) * BPT_NUM_FEATURES);
		arr[number - 1] = categoryToValue(outcome);

		return arr;
	};
	// may optimize with sprintf()
	std::string toString() {
		std::stringstream ss;
		for (int i = 0; i < numFeatures(); i++) {
			ss.fill('0');
			ss.width(4);
			ss << feature[i] << " ";
		}
		ss << categoryToValue(outcome);
		return ss.str();
	};
} Dataset;

typedef struct UniqueValues {
	std::vector<int> vals;
	int numVals = 0;
} UniqueValues;

typedef struct Result {
	std::string outcome = "";
	float probability = 0;
} Result;

typedef struct Partition {
	Record *true_branch, *false_branch;
	int true_branch_size = 0, false_branch_size = 0;
} Partition;

class Decision {
public:
	Decision();
	Decision(int ref, int feat);
	bool decide(Record trData);
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
	Node(NodeType type);
	bool isResult() { return type == RESULT; }
	Node* false_branch, *true_branch;
	virtual std::string toString() = 0;
	int getNum() { return numNode; }
protected:
	NodeType type;
	static int numOfNodes;
	int numNode;
};

class DecisionNode : tree::Node {
public:
	DecisionNode(Decision decision);
	Decision dec;
	std::string toString();
};

class ResultNode : tree::Node {
public:
	ResultNode(Result result);
	ResultNode(std::vector<Result> result);
	std::string toString();
	std::vector<Result> result;
	int numResults;
};

typedef struct NodeRefs {
	Node* node;
	int trueBranch = 0, falseBranch = 0, nodeNum = 0;
}NodeRefs;

void decisionNode(std::vector<tree::NodeRefs>& noderefs, std::string data);
void resultNode(std::vector<tree::NodeRefs>& noderefs, std::string data);
void buildTree(tree::Node*& tree, std::vector<tree::NodeRefs>& noderefs);
void addNode(tree::Node*& node, std::vector<tree::NodeRefs>& noderefs, int nodeNum);


}


#endif