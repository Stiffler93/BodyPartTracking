#ifndef TREE_TESTS
#define TREE_TESTS

#include "DecTree.h"

void testWithTrainingData(tree::Node * decisionTree);
void testWithTrainingDataWithoutRecord(tree::Node * decisionTree);
void testWithTestData(tree::Node* decisionTree, tree::Record* testData, int numTestData);

#endif TREE_TESTS