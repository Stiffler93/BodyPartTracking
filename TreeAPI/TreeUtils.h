#ifndef TREE_UTILS
#define TREE_UTILS

#include "DecTree.h"
#include <vector>
#include <fstream>

void printTree(tree::Node * node);

void saveTree(tree::Node * tree, std::ofstream& file);

void freeTree(tree::Node * node);

int treeDepth(tree::Node * node);

void normalizeTree(tree::Node*& origTree, tree::Node*& normalizedTree);

void findResult(tree::Node* node, tree::Record test, std::vector<tree::Result>& results);

bool getNextRecord(std::ifstream& dataset, tree::Record& record);

void trace(std::string trace);

void trace(const char * trc);

#endif TREE_UTILS