#include "opencv2\opencv.hpp"
#include "TreeConstants.h"
#include "DecTree.h"

using namespace cv;
using namespace std;

//This function returns an image where the subject is extracted
//from the original depth image
void getSubject(Mat& depImg, Mat& depImgSubject);

//Returns the Category according to the color value in the 
//RGB image
string getCategory(int red, int green, int blue);

//returns the color dependent to the category
Vec3b getBGR(string category);

//This feature extractor is supposed to classify centered pixels
//Therefore it uses the 4 edges from a virtual square projected
//into the scene, where the pixel is the center and a is the
//length of a side(in pixels)
void feature1(Mat& depImg, Mat& feature2, int offset);

//This feature extractor is supposed to classify top pixels
//It calculates the difference to a pixel density above with
//a certain offset specified by o
void feature2(Mat& depImg, Mat& feature2, int offset);

//This feature extractor is supposed to classify left pixels
//It calculates the difference to a pixel density to the left
//with a certain offset specified by o
void feature3(Mat& depImg, Mat& feature2, int offset);

//This feature extractor is supposed to classify left pixels
//It calculates the difference to a pixel density to the right
//with a certain offset specified by o
void feature4(Mat& depImg, Mat& feature2, int offset);

//This feature extractor is supposed to classify vertically
//thin areas(like arms, neck, head).It compares two pixels,
//left and right with the same offset o / 2 from the origin
void feature5(Mat& depImg, Mat& feature2, int offset);

//This feature extractor is supposed to classify horizontally
//thin areas(like arms).It compares two pixels,
//above and below with the same offset o / 2 from the origin
void feature6(Mat& depImg, Mat& feature2, int offset);

//This feature extractor is supposed to classify leftoutermost
//and / or uppermost parts.Therefore it compares two pixels,
//one left and one above with a specified offset o
void feature7(Mat& depImg, Mat& feature2, int offset);

//This feature extractor is supposed to classify rightoutermost
//and / or uppermost parts.Therefore it compares two pixels,
//one right and one above with a specified offset o
void feature8(Mat& depImg, Mat& feature2, int offset);

//This feature extractor is supposed to classify diagonally
//from top left to bottom right thin areas
void feature9(Mat& depImg, Mat& feature2, int offset);

//This feature extractor is supposed to classify diagonally
//from bottom left to top right thin areas
void feature10(Mat& depImg, Mat& feature2, int offset);

//Calculates all features for every pixel in the image and returns a matrix
//which size matches the image size and contains a Dataset (=feature collection)
//in every index. The indices of the Datasets correspond with the indices of
//the depending pixels.
void featurizeImage(Mat& depImg, tree::Dataset**& featureMatrix);