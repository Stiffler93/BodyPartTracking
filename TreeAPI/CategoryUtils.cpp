#include "CategoryUtils.h"

std::string categoryOfValue(int category) {
	switch (category) {
	case 0: return LEFT_SHOULDER;
	case 1: return RIGHT_SHOULDER;
	case 2: return HEAD;
	case 3: return NECK;
	case 4: return STERNUM;
	case 5: return OTHER;
	}
}

short categoryToValue(std::string category) {
	if (category == LEFT_SHOULDER)
		return 0;

	if (category == RIGHT_SHOULDER)
		return 1;

	if (category == HEAD)
		return 2;

	if (category == NECK)
		return 3;

	if (category == STERNUM)
		return 4;

	if (category == OTHER)
		return 5;
}