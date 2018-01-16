#include "ErgonomicEvaluation.h"
#include "3DWorldTransformations.h"

using namespace tree;
using namespace cv;

ergonomics::ErgonomicEvaluation* ergonomics::ErgonomicEvaluation::instance = NULL;

double ergonomics::ErgonomicEvaluation::distance(cv::Point3d p1, cv::Point3d p2)
{
	double xDist = p1.x - p2.x;
	double yDist = p1.y - p2.y;
	double zDist = p1.z - p2.z;

	return sqrt(pow(xDist, 2) + pow(yDist, 2) + pow(zDist, 2));
}

ergonomics::ErgonomicEvaluation::ErgonomicEvaluation()
{
	for (int area = 0; area < NUM_AREAS_OF_INTEREST; area++) {
		for (int counter = 0; counter < NUM_COUNTERS_PER_AREA; counter++) {
			counters[area][counter] = 100;
		}
	}
}

ergonomics::ErgonomicEvaluation::~ErgonomicEvaluation()
{
}

void ergonomics::ErgonomicEvaluation::process(tree::BodyPartLocations bpLocs)
{
}

Measurements ergonomics::ErgonomicEvaluation::classify(tree::BodyPartLocations bpLocs)
{
	world::CoordinateTransformator* transformator = world::CoordinateTransformator::getInstance();

	Point3d loc3D[LOC_NUMBER];
	
	for (int i = 0; i < LOC_NUMBER; i++) {
		BodyPartLocation bpLoc = bpLocs.locs[i];
		loc3D[i] = transformator->transformToWorldSpace(bpLoc.row, bpLoc.col, bpLoc.depth);
		//printf("loc3D[%d] = <%lf,%lf,%lf>\n", i, loc3D[i].x, loc3D[i].y, loc3D[i].z);
	}

	Measurements mm;

	mm.vals[MM_DIST_HEAD_TO_L_SHOULDER] = distance(loc3D[LOC_HEAD], loc3D[LOC_L_SHOULDER]);
	mm.vals[MM_DIST_HEAD_TO_R_SHOULDER] = distance(loc3D[LOC_HEAD], loc3D[LOC_R_SHOULDER]);
	mm.vals[MM_DIST_HEAD_TO_STERNUM] = distance(loc3D[LOC_HEAD], loc3D[LOC_STERNUM]);
	mm.vals[MM_DIST_STERNUM_TO_L_SHOULDER] = distance(loc3D[LOC_STERNUM], loc3D[LOC_L_SHOULDER]);
	mm.vals[MM_DIST_STERNUM_TO_R_SHOULDER] = distance(loc3D[LOC_STERNUM], loc3D[LOC_R_SHOULDER]);
	mm.vals[MM_DIST_NECK_TO_L_SHOULDER] = distance(loc3D[LOC_NECK], loc3D[LOC_L_SHOULDER]);
	mm.vals[MM_DIST_NECK_TO_R_SHOULDER] = distance(loc3D[LOC_NECK], loc3D[LOC_R_SHOULDER]);

	mm.vals[MM_OFF_HEAD_TO_NECK] = loc3D[LOC_HEAD].x - loc3D[LOC_NECK].x;
	mm.vals[MM_OFF_HEAD_TO_L_SHOULDER] = loc3D[LOC_HEAD].x - loc3D[LOC_L_SHOULDER].x;
	mm.vals[MM_OFF_HEAD_TO_R_SHOULDER] = loc3D[LOC_HEAD].x - loc3D[LOC_R_SHOULDER].x;
	mm.vals[MM_OFF_STERNUM_TO_L_SHOULDER] = loc3D[LOC_STERNUM].x - loc3D[LOC_L_SHOULDER].x;
	mm.vals[MM_OFF_STERNUM_TO_R_SHOULDER] = loc3D[LOC_STERNUM].x - loc3D[LOC_R_SHOULDER].x;
	
	mm.vals[MM_OFF_DEP_HEAD_TO_NECK] = loc3D[LOC_HEAD].z - loc3D[LOC_NECK].z;
	mm.vals[MM_OFF_DEP_NECK_TO_STERNUM] = loc3D[LOC_NECK].z - loc3D[LOC_STERNUM].z;
	mm.vals[MM_OFF_DEP_HEAD_TO_STERNUM] = loc3D[LOC_HEAD].z - loc3D[LOC_STERNUM].z;
	mm.vals[MM_OFF_DEP_L_SHOULDER_TO_R_SHOULDER] = loc3D[LOC_L_SHOULDER].z - loc3D[LOC_R_SHOULDER].z;
	mm.vals[MM_OFF_DEP_L_SHOULDER_TO_STERNUM] = loc3D[LOC_L_SHOULDER].z - loc3D[LOC_STERNUM].z;
	mm.vals[MM_OFF_DEP_R_SHOULDER_TO_STERNUM] = loc3D[LOC_R_SHOULDER].z - loc3D[LOC_STERNUM].z;

	//printf("%s\r", mm.toString().c_str());
	//printf("Dist_H-LS = %3.5lf\r", distance_head_to_left_shoulder);
	//printf("Head_Vertical_Alignment = %3.5lf\r", mm.vals[MM_DIST_HEAD_TO_L_SHOULDER] - mm.vals[MM_DIST_HEAD_TO_R_SHOULDER]);
	
	return mm;
}
