#include "Ergonomics.h"
#include "OpenNI.h"
#include "opencv2\opencv.hpp"

#include <string>
#include <vector>

typedef int Stream;

#define DEPTH_STREAM 0
#define COLOR_STREAM 1

#define MAT_ROWS 240
#define MAT_COLS 320

#define WINDOW_DEPTH_RECORD "Depth Record"
#define WINDOW_COLOR_RECORD "Color Record"
#define WINDOW_DEPTH "Depth Stream"
#define WINDOW_COLOR "Color Stream"

#define START_IMAGE_WITH_INDEX 49

#define RECORD (int)'r'
#define FREE (int) 'f'
#define SAVE (int) 's'

#define IMG_FOLDER "C:\\Users\\Stefan\\Desktop\\Master\\Masterarbeit\\Programme\\BPT_TrainingImages\\"

namespace util {

	class ImageSaver {
	public:
		static void save(cv::Mat image, std::string name = "", int num = 0);

	private:
		static int numImage;
		ImageSaver();
		ImageSaver(ImageSaver& imageSaver);
		~ImageSaver();
	};

	class ImageRecorder
	{
	public:
		ImageRecorder(openni::Device& device, openni::VideoStream** streams);
		~ImageRecorder();

		void addDepthOperation(ergonomics::ImageOperation imgOp);
		void addColorOperation(ergonomics::ImageOperation imgOp);
		void addDepthListener(ergonomics::ImageListener listener);
		void addColorListener(ergonomics::ImageListener listener);

		void run();

	private:
		openni::Device& device;
		openni::VideoStream** streams;
		openni::VideoFrameRef colorFrame, depthFrame;
		cv::Mat depthMat, colorMat, depthMatRecorded, colorMatRecorded;

		int counter = 0;
		int numImage = 1;

		std::vector<ergonomics::ImageOperation> depthOperations;
		std::vector<ergonomics::ImageOperation> colorOperations;

		std::vector<ergonomics::ImageListener> depthListeners;
		std::vector<ergonomics::ImageListener> colorListeners;

		int img = (int) START_IMAGE_WITH_INDEX;

		void createWindows();
		void initCV();
		void readStreams();
		void readFrame(openni::VideoStream& stream, openni::VideoFrameRef& frame);
		void triggerListener(cv::Mat& img, Stream stream);
		void processImage(cv::Mat& img, Stream stream, const char* window);

		void handleKey(int key);
		void save();
	};

}

