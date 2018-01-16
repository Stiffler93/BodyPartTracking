#include "opencv2\opencv.hpp"
#include "TreeConstants.h"
#include "DecTree.h"

//This function returns an image where the subject is extracted
//from the original depth image
void getSubject(cv::Mat& depImg, cv::Mat& depImgSubject);

//This function takes an image an calculates the horizontal integral
//where every pixel != 0 is counted
void getHorizontalIntegral(cv::Mat& image, cv::Mat& horizIntegral);

//This function takes an image an calculates the horizontal integral
//where every pixel != 0 is counted
void getVerticalIntegral(cv::Mat& image, cv::Mat& vertIntegral);

//This function calculates the integral image corresponding to the
//input image
void getIntegral(cv::Mat& image, cv::Mat& integral);

//Returns the Category according to the color value in the 
//RGB image
std::string getCategory(int red, int green, int blue);

//returns the color dependent to the category
cv::Vec3b getBGR(std::string category);

//This feature extractor is supposed to classify centered pixels
//Therefore it uses the 4 edges from a virtual square projected
//into the scene, where the pixel is the center and a is the
//length of a side(in pixels)
void feature1(cv::Mat& depImg, cv::Mat& feature1, cv::Mat& depthImg, int offset);

//This feature extractor is supposed to classify top pixels
//It calculates the difference to a pixel density above with
//a certain offset specified by o
void feature2(cv::Mat& depImg, cv::Mat& feature2, cv::Mat& depthImg, int offset);

//This feature extractor is supposed to classify left pixels
//It calculates the difference to a pixel density to the left
//with a certain offset specified by o
void feature3(cv::Mat& depImg, cv::Mat& feature3, cv::Mat& depthImg, int offset);

//This feature extractor is supposed to classify left pixels
//It calculates the difference to a pixel density to the right
//with a certain offset specified by o
void feature4(cv::Mat& depImg, cv::Mat& feature4, cv::Mat& depthImg, int offset);

//This feature extractor is supposed to classify vertically
//thin areas(like arms, neck, head).It compares two pixels,
//left and right with the same offset o / 2 from the origin
void feature5(cv::Mat& depImg, cv::Mat& feature5, cv::Mat& depthImg, int offset);

//This feature extractor is supposed to classify horizontally
//thin areas(like arms).It compares two pixels,
//above and below with the same offset o / 2 from the origin
void feature6(cv::Mat& depImg, cv::Mat& feature6, cv::Mat& depthImg, int offset);

//This feature extractor is supposed to classify leftoutermost
//and / or uppermost parts.Therefore it compares two pixels,
//one left and one above with a specified offset o
void feature7(cv::Mat& depImg, cv::Mat& feature7, cv::Mat& depthImg, int offset);

//This feature extractor is supposed to classify rightoutermost
//and / or uppermost parts.Therefore it compares two pixels,
//one right and one above with a specified offset o
void feature8(cv::Mat& depImg, cv::Mat& feature8, cv::Mat& depthImg, int offset);

//This feature extractor is supposed to classify diagonally
//from top left to bottom right thin areas
void feature9(cv::Mat& depImg, cv::Mat& feature9, cv::Mat& depthImg, int offset);

//This feature extractor is supposed to classify diagonally
//from bottom left to top right thin areas
void feature10(cv::Mat& depImg, cv::Mat& feature10, cv::Mat& depthImg, int offset);

//This feature extractor is supposed to classify horizontally centered
//pixels. It classifies the proportion of pixels to the left and to the right
void feature11(cv::Mat& depImg, cv::Mat& feature11, int offset, cv::Mat& horizIntegral = cv::Mat());

//This feature extractor is supposed to classify vertically centered
//pixels. It classifies the proportion of pixels to the top and to the bottom
void feature12(cv::Mat& depImg, cv::Mat& feature12, int offset, cv::Mat& vertIntegral = cv::Mat());

//This feature extractor is supposed to classify thin areas, as well as borders and 
//centers as well. It depends on how high the offset is set. In general, it calculates
//the number of pixels belonging to the subject in a square with size offset * 2
void feature13(cv::Mat& depImg, cv::Mat& feature13, int offset, cv::Mat& integral = cv::Mat());

//Calculates all features for every pixel in the image and returns a matrix
//which size matches the image size and contains a Dataset (=feature collection)
//in every index. The indices of the Datasets correspond with the indices of
//the depending pixels.
void featurizeImage(cv::Mat& depImg, tree::Dataset**& featureMatrix);