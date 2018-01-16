#include "opencv2\opencv.hpp"
#include "Features.hpp"

#define IMAGE "C:/Users/Stefan/Desktop/Master/Masterarbeit/Programme/BPT_TrainingImages/StraightPostures/depth/147.png"

int main_(int argc, char** argv) {
	cv::Mat img = cv::imread(IMAGE, CV_LOAD_IMAGE_ANYDEPTH);
	
	cv::Mat subject = cv::Mat(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	getSubject(img, subject);

	img.setTo(0, subject == 0);

	int key = 0;
	while (key != 27) {
		imshow("Depth Image", img);
		key = cv::waitKey(10);
	}

	return 0;
}