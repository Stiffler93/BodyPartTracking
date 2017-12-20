#include "DecTree.h"
#include "TreeSettings.h"
#include <fstream>
#include <string>
#include "TreeConstants.h"
#include "TreeUtils.h"
#include "opencv2\opencv.hpp"
#include "ImageRecorder.h"
#include "Features.hpp"
#include "Tests.h"

using namespace tree;

typedef struct NodeRefs {
	tree::Node* node;
	int trueBranch = 0, falseBranch = 0, nodeNum = 0;
}NodeRefs;

void decisionNode(vector<NodeRefs> *noderefs, string data);
void resultNode(vector<NodeRefs> *noderefs, string data);
void buildTree(tree::Node*& tree, vector<NodeRefs> *noderefs);
void addNode(tree::Node*& node, vector<NodeRefs> *noderefs, int nodeNum);
void classification(Mat& image, Mat& classifiedMat, tree::Dataset**& featureMatrix, tree::Node* decisionTree);
void realTest(tree::Node* decisionTree);

vector<int> referencedNodes;
vector<int> nodeNums;

int main(int argc, char** argv) 
{
	vector<NodeRefs> nodes;

	ifstream treeFile(treeFile());
	string data = "";
	while (treeFile >> data) {
		if (data.empty())
			break;

		if (data.substr(0, 1) == "D") {
			decisionNode(&nodes, data);
		}
		else if (data.substr(0, 1) == "R") {
			resultNode(&nodes, data);
		}
		else {
			throw exception("Unrecognized Tree Node! Only DecisionNodes [D] and ResultNodes [R] are valid!");
		}
	}

	trace("Start Debug:");

	tree::Node* tree = NULL;
	buildTree(tree, &nodes);

	printf("Tree successfully reconstructed!\n");
	printf("Start Decision Tree Test: \n");

	//testWithTrainingData(tree);
	realTest(tree);

	ofstream tree_file(tree::treeFile());
	saveTree(tree, tree_file);
	tree_file.close();

	string s;
	cin >> s;

	return 0;
}

void decisionNode(vector<NodeRefs> *noderefs, string data)
{
	string tmp;
	stringstream values;
	int nodeNum = 0, refVal = 0, feature = 0, trueBranch = 0, falseBranch;
	tmp = data.substr(2, data.length() - 3);

	for (int i = 0; i < tmp.length(); i++) {
		char c = tmp.at(i);
		

		if (c == ',')
			c = ' ';

		values << c;
	}

	values >> nodeNum >> refVal >> feature >> trueBranch >> falseBranch;

	if (find(nodeNums.begin(), nodeNums.end(), nodeNum) != nodeNums.end()) {
		trace("nodeNum " + to_string(nodeNum) + " was already handled before!");
	}

	if (find(referencedNodes.begin(), referencedNodes.end(), trueBranch) != referencedNodes.end()) {
		trace("Node " + to_string(trueBranch) + " is referenced multiple times");
	}

	if (find(referencedNodes.begin(), referencedNodes.end(), falseBranch) != referencedNodes.end()) {
		trace("Node " + to_string(falseBranch) + " is referenced multiple times");
	}

	nodeNums.push_back(nodeNum);
	referencedNodes.push_back(trueBranch);
	referencedNodes.push_back(falseBranch);

	NodeRefs ref;
	ref.trueBranch = trueBranch;
	ref.falseBranch = falseBranch;
	ref.nodeNum = nodeNum;
	ref.node = (tree::Node*) new DecisionNode(Decision(refVal, feature));

	noderefs->push_back(ref);
}

void resultNode(vector<NodeRefs> *noderefs, string data)
{
	string tmp;
	stringstream values;

	char c;
	for (int i = 1; i < data.length(); i++) {
		c = data.at(i);

		if (c == 'C' || c == '(' || c == ')')
			continue;
		
		if (c == ',')
			c = ' ';

		values << c;
	}

	string numNode;
	string outcome;
	string probability;
	vector<Result> results;

	values >> numNode;
	while (values >> outcome >> probability) {
		Result r;
		r.outcome = outcome;
		r.probability = stof(probability.c_str());
		results.push_back(r);
	}
	
	NodeRefs ref;
	ref.trueBranch = 0;
	ref.falseBranch = 0;
	ref.nodeNum = stoi(numNode);
	ref.node = (tree::Node*) new ResultNode(results);

	noderefs->push_back(ref);
}

void buildTree(tree::Node*& tree, vector<NodeRefs>* noderefs)
{
	//printf("build tree: \n");
	NodeRefs& ref = noderefs->at(0);
	if (ref.nodeNum != 1) {
		trace("Wrong nodeNum! Was supposed to be 1, but is " + to_string(ref.nodeNum));
	}

	tree = ref.node;
	ref.node = NULL;
	if (ref.trueBranch != 0)
		addNode(tree->true_branch, noderefs, ref.trueBranch);
	if (ref.falseBranch != 0)
		addNode(tree->false_branch, noderefs, ref.falseBranch);

	for (NodeRefs r : *noderefs) {
		if (r.node != NULL) {
			trace("Node " + to_string(r.nodeNum) + " points to Node!");
		}
	}
}

void addNode(tree::Node*& node, vector<NodeRefs>* noderefs, int nodeNum)
{
	//printf("Add Node with num %d\n", nodeNum - 1);
	NodeRefs& ref = noderefs->at(nodeNum - 1);
	if (ref.nodeNum != nodeNum) {
		trace("Correct node not immediately found!");
		int diff = nodeNum - ref.nodeNum;
		ref = noderefs->at(ref.nodeNum + diff);

		if (ref.nodeNum != nodeNum) {
			trace("Node not found at all!");
			return;
		}
	}

	if (ref.nodeNum != nodeNum) {
		trace("Wrong nodeNum! Was supposed to be " + to_string(nodeNum) + ", but is " + to_string(ref.nodeNum));
	}

	node = ref.node;
	ref.node = NULL;

	if (ref.trueBranch != 0)
		addNode(node->true_branch, noderefs, ref.trueBranch);
	if (ref.falseBranch != 0)
		addNode(node->false_branch, noderefs, ref.falseBranch);
}

//void test(tree::Node * decisionTree)
//{
//	string dataset = datasetFile();
//	ifstream features(dataset);
//
//	int feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10;
//	string category;
//	int total = 0, correctClass = 0, falseClass = 0;
//	vector<Result> results;
//
//	while (features >> feat1 >> feat2 >> feat3 >> feat4 >> feat5 >> feat6 >> feat7 >> feat8 >> feat9 >> feat10 >> category) {
//		Dataset set;
//		set.feature[0] = feat1;
//		set.feature[1] = feat2;
//		set.feature[2] = feat3;
//		set.feature[3] = feat4;
//		set.feature[4] = feat5;
//		set.feature[5] = feat6;
//		set.feature[6] = feat7;
//		set.feature[7] = feat8;
//		set.feature[8] = feat9;
//		set.feature[9] = feat10;
//		set.outcome = category;
//
//		total++;
//
//		results.clear();
//		findResult(decisionTree, set, results);
//
//		if (results.size() == 0) {
//			falseClass++;
//			//printf("No Result returned\n");
//		}
//		else {
//			if (results.size() == 1) {
//				Result res = results.at(0);
//				if (res.outcome == category) {
//					correctClass++;
//				}
//				else {
//					//printf("Wrong result.\n");
//					falseClass++;
//				}
//			}
//			else {
//				bool found = false;
//				float prob = 0;
//
//				for(Result res : results) 
//					if (res.outcome == category) {
//						found = true;
//						prob = res.probability;
//					}
//
//				if (found) {
//					correctClass++;
//					//printf("Found, but low probability: >%f<\n", prob);
//				}
//				else {
//					falseClass++;
//				}
//			}
//		}
//	}
//
//	printf("\n\nTest finished.\n");
//	printf("Tested a total of %d Datasets.\n", total);
//	printf("Correct: >%d<, False: >%d<\n", correctClass, falseClass);
//	printf("Correctly classified %lf percent.\n", (float)correctClass / (float)total * 100);
//}

void realTest(tree::Node * decisionTree)
{
	openni::Status statOpenNI;
	printf("OpenNI initialization...\n");
	statOpenNI = OpenNI::initialize();
	if (statOpenNI != openni::Status::STATUS_OK) {
		puts("OpenNI initialization failed!");
		return;
	}

	puts("Asus Xtion Pro initialization...");
	Device device;
	if (device.open(openni::ANY_DEVICE) != 0)
	{
		puts("Device not found !");
		puts("Abort test");
		OpenNI::shutdown();
		return;
	}
	puts("Asus Xtion Pro opened");

	VideoStream depth;
	depth.create(device, SENSOR_DEPTH);
	depth.start();

	VideoMode paramvideo;
	paramvideo.setResolution(MAX_COL, MAX_ROW);
	paramvideo.setFps(FPS);
	paramvideo.setPixelFormat(PIXEL_FORMAT_DEPTH_100_UM);

	depth.setVideoMode(paramvideo);

	VideoStream** stream = new VideoStream*[1];
	stream[0] = &depth;
	puts("Kinect initialization completed");

	puts("Continue? (y/n)");
	string s;
	cin >> s;

	if (s != "y") {
		puts("Shutdown OpenNI");
		depth.stop();
		depth.destroy();
		device.close();
		OpenNI::shutdown();
		return;
	}

	util::ImageRecorder recorder(device, stream, decisionTree, &classification);
	recorder.run();

	printf("Release OpenNI resources.");
	depth.stop();
	depth.destroy();
	printf(".");
	device.close();
	printf(".");
	OpenNI::shutdown();
	printf("done.\n");
}

void classification(Mat& image, Mat& classifiedMat, tree::Dataset**& featureMatrix, tree::Node* decisionTree) {
	classifiedMat = 0;
	
	featurizeImage(image, featureMatrix);
	Vec3b color;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			tree::Dataset set = featureMatrix[row][col];

			if (set.outcome == OTHER) {
				vector<Result> result;
				findResult(decisionTree, set, result);

				// atm only one result is returned
				Result res = result.at(0);
				color = getBGR(res.outcome);
			}
			else {
				color = { 0,0,0 };
			}
			
			classifiedMat.at<Vec3b>(row, col) = color;
		}
	}
}
