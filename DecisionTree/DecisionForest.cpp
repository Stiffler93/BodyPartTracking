#include "DecisionForest.h"
#include "TreeUtils.h"
#include <fstream>
#include <map>

using std::vector;
using std::string;

tree::DecisionForest::DecisionForest() {};

tree::DecisionForest::DecisionForest(vector<string> treeFiles) {
	for (string tree : treeFiles) {
		parseTree(tree);
	}
}

string tree::DecisionForest::classify(Dataset set)
{
	std::map<string, double> overallResult;

	for (tree::Node* node : trees) {
		vector<Result> results;
		findResult(node, set, results);

		for (Result r : results) {
			std::map<string, double>::iterator it = overallResult.find(r.outcome);
			if (it == overallResult.end()) {
				overallResult.insert(std::pair<string, double>(r.outcome, r.probability));
			}
			else {
				it->second += r.probability;
			}
		}
	}

	string outcome;
	double probability = 0;
	for (std::pair<string, double> entry : overallResult) {
		if (entry.second > probability) {
			outcome = entry.first;
			probability = entry.second;
		}
	}

	//if(probability >= trees.size() / 3 * 2) 
	//	return outcome;

	//return OTHER;
	return outcome;
}

void tree::DecisionForest::parseTree(std::string treeFilePath)
{
	std::vector<tree::NodeRefs> nodes;

	std::ifstream treeFile(treeFilePath);
	string data = "";
	while (treeFile >> data) {
		if (data.empty())
			break;

		if (data.substr(0, 1) == "D") {
			decisionNode(nodes, data);
		}
		else if (data.substr(0, 1) == "R") {
			resultNode(nodes, data);
		}
		else {
			throw std::exception("Unrecognized Tree Node! Only DecisionNodes [D] and ResultNodes [R] are valid!");
		}
	}

	tree::Node* tree = NULL;
	buildTree(tree, nodes);

	trees.push_back(tree);

	printf("Tree <%s> successfully reconstructed!\n", treeFilePath.c_str());
}
