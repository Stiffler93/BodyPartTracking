#include "OpenNI.h"
#include "opencv2\opencv.hpp"

namespace world {

	class CoordinateTransformator {
	private:
		CoordinateTransformator();
		static CoordinateTransformator* instance;

		openni::VideoStream* stream = NULL;
	public:
		~CoordinateTransformator();
		static CoordinateTransformator* getInstance() {
			if (instance == NULL) {
				instance = new CoordinateTransformator();
			}
			return instance;
		}
		void init(openni::VideoStream* stream);
		cv::Point3d transformToWorldSpace(int row, int col, ushort depth);
	};
}