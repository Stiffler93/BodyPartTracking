#ifndef DEC_FOREST
#define DEC_FOREST

#include "DecTree.h"
#include <vector>
#include <string>

namespace tree {

	class DecisionForest {
	private:
		std::vector<Node*> trees;
	public:
		DecisionForest();
		DecisionForest(std::vector<std::string> treeFiles);
		std::string classify(Record set);
		void parseTree(std::string tree);
	};
}

#endif