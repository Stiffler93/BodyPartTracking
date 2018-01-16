#ifndef TREE_UTILS
#define TREE_UTILS

#include "DecTree.h"
#include <vector>
#include <fstream>

void printTree(tree::Node * node);

void saveTree(tree::Node * tree, std::ofstream& file);

void freeTree(tree::Node * node);

void findResult(tree::Node* node, tree::Dataset test, std::vector<tree::Result>& results);

bool getNextRecord(std::ifstream& dataset, tree::Dataset& record);

void trace(std::string trace);

void trace(const char * trc);

#endif TREE_UTILS