#include "TreeUtils.h"

void printTree(tree::Node * node)
{
	printf("%s\n", node->toString().c_str());
	if (node->true_branch != NULL)
		printTree(node->true_branch);
	if (node->false_branch != NULL)
		printTree(node->false_branch);
}

void saveTree(tree::Node * tree, ofstream& file)
{
	file << tree->toString() << endl;

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

void findResult(tree::Node* node, tree::Dataset test, std::vector<tree::Result>& results) {
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
		throw exception("No Result found!!!\n");
	}
}

void trace(string trace)
{
	static ofstream traceFile(tree::debugFile());
	if (tree::isTraceActive())
		traceFile << trace << endl;
}

void trace(const char * trc)
{
	trace(string(trc));
}