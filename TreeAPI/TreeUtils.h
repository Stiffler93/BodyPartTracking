#ifndef TREE_UTILS
#define TREE_UTILS

#include "DecTree.h"
#include <vector>

using namespace tree;

void printTree(Node * node)
{
	printf("%s\n", node->toString().c_str());
	if (node->true_branch != NULL)
		printTree(node->true_branch);
	if (node->false_branch != NULL)
		printTree(node->false_branch);
}

void findResult(Node* node, Dataset test, std::vector<Result>& results) {
	while (!node->isResult()) {
		if (((DecisionNode*)node)->dec.decide(test)) {
			node = node->true_branch;
		}
		else {
			node = node->false_branch;
		}
	}

	if (node->isResult()) {
		results = ((ResultNode*)node)->result;
	}
	else {
		printf("Fatal Error! No Result found!!!\n");
		throw exception("No Result found!!!\n");
	}
}

#endif TREE_UTILS