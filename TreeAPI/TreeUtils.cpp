#include "TreeUtils.h"
#include <fstream>
#include <sstream>
#include <algorithm>

void printTree(tree::Node * node)
{
	printf("%s\n", node->toString().c_str());
	if (node->true_branch != NULL)
		printTree(node->true_branch);
	if (node->false_branch != NULL)
		printTree(node->false_branch);
}

void saveTree(tree::Node * tree, std::ofstream& file)
{
	file << tree->toString() << std::endl;

	if (tree->true_branch != NULL)
		saveTree(tree->true_branch, file);
	if (tree->false_branch != NULL)
		saveTree(tree->false_branch, file);
}

void freeTree(tree::Node * node) {
	if (node->true_branch != NULL)
		freeTree(node->true_branch);
	if (node->false_branch != NULL)
		freeTree(node->false_branch);

	delete node;
}

int treeDepth(tree::Node * node)
{
	int trueDepth = 0, falseDepth = 0;

	if (node->true_branch != NULL)
		trueDepth = treeDepth(node->true_branch);

	if (node->false_branch != NULL)
		falseDepth = treeDepth(node->false_branch);

	return std::max(trueDepth, falseDepth) + 1;
}

void normTree(tree::Node * origTree, tree::Node * normalizedTree) 
{
	if (origTree->true_branch != NULL) {
		if (origTree->true_branch->isResult()) {
			normalizedTree->true_branch = (tree::Node*) new tree::ResultNode(((tree::ResultNode*) origTree->true_branch)->result);
		}
		else {
			normalizedTree->true_branch = (tree::Node*) new tree::DecisionNode(((tree::DecisionNode*) origTree->true_branch)->dec);
			normTree(origTree->true_branch, normalizedTree->true_branch);
		}
	}

	if (origTree->false_branch != NULL) {
		if (origTree->false_branch->isResult()) {
			normalizedTree->false_branch = (tree::Node*) new tree::ResultNode(((tree::ResultNode*) origTree->false_branch)->result);
		}
		else {
			normalizedTree->false_branch = (tree::Node*) new tree::DecisionNode(((tree::DecisionNode*) origTree->false_branch)->dec);
			normTree(origTree->false_branch, normalizedTree->false_branch);
		}
	}
}

void normalizeTree(tree::Node*& origTree, tree::Node*& normalizedTree)
{
	if (origTree != NULL) {
		tree::Node::numOfNodes = 0;
		normalizedTree = (tree::Node*) new tree::DecisionNode(((tree::DecisionNode*) origTree)->dec);
		normTree(origTree, normalizedTree);
	}
}

void findResult(tree::Node* node, tree::Record test, std::vector<tree::Result>& results) {
	while (!node->isResult()) {
		if (((tree::DecisionNode*)node)->dec.decide(test)) {
			node = node->true_branch;
		}
		else {
			node = node->false_branch;
		}
	}

	if (node->isResult()) {
		results = ((tree::ResultNode*)node)->result;
	}
	else {
		printf("Fatal Error! No Result found!!!\n");
		throw std::exception("No Result found!!!\n");
	}
}

bool getNextRecord(std::ifstream& dataset, tree::Record& record)
{
	std::string s;
	std::getline(dataset, s);

	if (s.empty())
		return false;

	std::stringstream ss;

	ss << s;

	int value = 0;
	int feat = 0;
	while (ss >> value) {
		if (feat < tree::numFeatures()) {
			record.feature[feat] = value;
		}
		feat++;
	}

	if (feat <= tree::numFeatures())
		throw std::exception("getNextRecord didn't read all values!");
	
	record.outcome = categoryOfValue(value);
	
	return true;
}

void trace(std::string trace)
{
	static std::ofstream traceFile(tree::debugFile());
	if (tree::isTraceActive())
		traceFile << trace << std::endl;
}

void trace(const char * trc)
{
	trace(std::string(trc));
}