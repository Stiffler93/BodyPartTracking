#ifndef TREE_UTILS
#define TREE_UTILS

#include "DecTree.h"
#include "TreeSettings.h"
#include <vector>
#include "TreeConstants.h"

void printTree(tree::Node * node);

void saveTree(tree::Node * tree, ofstream& file);

void freeTree(tree::Node * node);

void findResult(tree::Node* node, tree::Dataset test, std::vector<tree::Result>& results);

void trace(string trace);

void trace(const char * trc);

#endif TREE_UTILS