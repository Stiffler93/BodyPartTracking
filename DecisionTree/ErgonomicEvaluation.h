#include "TreeConstants.h"
#include "BodyPartDetector.h"
#include "opencv2\opencv.hpp"
#include <string>
#include <sstream>

#define MM_DIST_HEAD_TO_L_SHOULDER 0
#define MM_DIST_HEAD_TO_R_SHOULDER 1
#define MM_DIST_HEAD_TO_STERNUM 2
#define MM_DIST_STERNUM_TO_L_SHOULDER 3
#define MM_DIST_STERNUM_TO_R_SHOULDER 4
#define MM_DIST_NECK_TO_L_SHOULDER 5
#define MM_DIST_NECK_TO_R_SHOULDER 6
#define MM_OFF_HEAD_TO_NECK 7
#define MM_OFF_HEAD_TO_L_SHOULDER 8
#define MM_OFF_HEAD_TO_R_SHOULDER 9
#define MM_OFF_STERNUM_TO_L_SHOULDER 10
#define MM_OFF_STERNUM_TO_R_SHOULDER 11
#define MM_OFF_DEP_HEAD_TO_NECK 12
#define MM_OFF_DEP_NECK_TO_STERNUM 13
#define MM_OFF_DEP_HEAD_TO_STERNUM 14
#define MM_OFF_DEP_L_SHOULDER_TO_R_SHOULDER 15
#define MM_OFF_DEP_L_SHOULDER_TO_STERNUM 16
#define MM_OFF_DEP_R_SHOULDER_TO_STERNUM 17
#define MM_TOTAL_NUM 18

#define ERG_AREA_NECK 0
#define ERG_AREA_UPPER_BODY 1
#define ERG_AREA_TOTAL_NUM 2
#define ERG_VALIDATION_LEFT 0
#define ERG_VALIDATION_BACK 1
#define ERG_VALIDATION_RIGHT 2
#define ERG_VALIDATION_FRONT 3
#define ERG_VALIDATION_TOTAL_NUM 4
namespace ergonomics {

	enum Strain {
		NONE, LIGHT, MEDIUM
	};

	typedef struct Measurements {
		double vals[MM_TOTAL_NUM];
		std::string toString() {
			std::stringstream ss;
			ss << "M(";
			ss << vals[0];
			for (int i = 1; i < MM_TOTAL_NUM; i++) {
				ss << "," << vals[i];
			}
			ss << ")";

			return ss.str();
		}
		void ofString(std::string s) {
			std::stringstream ss;
			if (s.at(0) != 'M') {
				printf("String were no Measurements! -> RETURN\n");
				return;
			}

			for (int c = 1; c < s.length(); c++) {
				char ch = s.at(c);
				if (ch == '(' || ch == ')')
					continue;

				if (ch == ',')
					ch = ' ';

				ss << ch;
			}

			int num_measurements_found = 0;
			int num;
			while (ss >> num) {
				vals[num_measurements_found++] = num;
			}

			if (num_measurements_found != MM_TOTAL_NUM) {
				printf("MM_TOTAL_NUM is %d, but only %d Measurements were found!\n", MM_TOTAL_NUM, num_measurements_found);
			}
		}
	} Measurements;

	typedef struct Strains {
		Strain strains[ERG_AREA_TOTAL_NUM][ERG_VALIDATION_TOTAL_NUM];
		std::string toString() {
			std::stringstream ss;
			ss << "S(";

			for (int area = 0; area < ERG_AREA_TOTAL_NUM; area++) {
				if (area == 0) {
					ss << "P(";
				}
				else {
					ss << ", P(";
				}
				ss << strains[area][0];
				for (int vals = 1; vals < ERG_VALIDATION_TOTAL_NUM; vals++) {
					ss << "," << strains[area][vals];
				}
				ss << ")";
			}
			ss << ")";
		}

		void ofString(std::string s) {
			if (s.at(0) != 'S') {
				printf("String were no Strains! -> RETURN\n");
				return;
			}

			int parts = 0, val;
			for (int c = 1; c < s.length(); c++) {
				char ch = s.at(c);

				if (ch == '(' || ch == ')' || ch == ',')
					continue;

				if (ch == 'P') {
					parts++;
					val = 0;
				}

				if ((int) ch >= (int)'0' && (int)ch <= (int)'2') {
					strains[parts - 1][val] = Strain(atoi(&ch));
				}
			}
		}
	} Strains;

	typedef struct Dataset {
		Measurements mm;
		Strains strains;
		std::string toString() {
			std::stringstream ss;
			ss << "MS(" << mm.toString() << ";" << strains.toString() << ")";
			return ss.str();
		}
		void ofString(std::string s) {
			if (s.substr(0, 2) != "MS") {
				printf("string were no Measurement and Strains! -> RETURN\n");
			}

			std::string str = s.substr(3, s.length() - 4);
			int posSemic = str.find(';', 0);
			std::string sMM = str.substr(0, posSemic - 1);
			std::string sSTR = str.substr(posSemic + 1, str.length() - posSemic - 2);

			mm.ofString(sMM);
			strains.ofString(sSTR);
		}
	} Dataset;

	class ErgonomicEvaluation {
	private:
		static ErgonomicEvaluation* instance;
		float counters[NUM_AREAS_OF_INTEREST][NUM_COUNTERS_PER_AREA];
		double distance(cv::Point3d p1, cv::Point3d p2);
	public:
		ErgonomicEvaluation();
		~ErgonomicEvaluation();
		static ErgonomicEvaluation getInstance() {
			if (instance == NULL) {
				instance = new ErgonomicEvaluation();
			}

			return *instance;
		}
		void process(tree::BodyPartLocations bpLocs);
		Measurements classify(tree::BodyPartLocations bpLocs);
	};
}