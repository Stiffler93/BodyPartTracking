
#ifndef TREE_CONSTS
#define TREE_CONSTS

#include <string>

using namespace std;

const static unsigned short MAX_VAL = 65535;
const static int MAX_ROW = 240;
const static int MAX_COL = 320;
const static int FPS = 30;
const static int DEPTH_IMAGE = 2; //= CV_16U
const static int MIN_DIST = 5000;
const static int NUM_CATEGORIES = 6; // NONE not counted!
const static int NORM_FACTOR = 1000; // max value of pixels in normalized image

const static string LEFT_SHOULDER = "LeftShoulder";
const static string RIGHT_SHOULDER = "RightShoulder";
const static string HEAD = "Head";
const static string NECK = "Neck";
const static string STERNUM = "Sternum";
const static string OTHER = "Other";
const static string NONE = "None";

#endif