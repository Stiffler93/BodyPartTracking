#ifndef ERGONOMICS_TYPES
#define ERGONOMICS_TYPES

typedef struct BodyPart {
	int row = 0;
	int col = 0;
} BODY_PART;

typedef struct BodyParts {
	BodyPart cervicalSpineTop;
	BodyPart cervicalSpineBottom;
	BodyPart sternum;
	BodyPart leftShoulder;
	BodyPart rightShoulder;
} BODY_PARTS;

#endif 