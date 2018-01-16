#include <OpenNI.h>

class FrameReader
{
public:
	FrameReader(openni::Device& device, openni::VideoStream& stream);
	virtual ~FrameReader();

protected:
	virtual void displayCV();
	virtual void glutKey(unsigned char key, int, int);

	openni::Device& device;
	openni::VideoStream& stream;
	openni::VideoStream** streams;

	openni::VideoFrameRef frame;

private:
	int streamWidth;
	int streamHeight;

	int texMapWidth;
	int texMapHeight;
	openni::RGB888Pixel* texMap;

	bool recordFrame;

	void printFrame();

};

