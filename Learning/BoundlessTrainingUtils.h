#pragma once

#include "TreeSettings.h"
#include "TreeUtils.h"
#include <string>

#define BITS_VALUE 8
#define MOVE_FLAG sizeof(char) * BITS_VALUE - 1
#define FLAG_PROCESSED UCHAR_MAX - CHAR_MAX
#define MASK_VALUE CHAR_MAX
#define MASK_SUBSET (ULLONG_MAX - UCHAR_MAX)
#define DONE 37

typedef struct RecordValuePair {
	char record[tree::BPT_DATASET_RECORD_CHARS];
	char dummy;
	char value[tree::BPT_DATASET_VALUE_CHARS];
	char dummy2;
}RecordValuePair;

typedef struct PlainRecord {
	RecordValuePair feature[tree::BPT_NUM_FEATURES];
}PlainRecord;

typedef struct MapRecord {
	char record[tree::BPT_DATASET_RECORD_CHARS];
	char dummy;
	char category;
	char dummy2;
}MapRecord;

class Metadata;

class FeatureStats {
public:
	FeatureStats();
	FeatureStats(int feature, Metadata* metadata);
	void nextValue(short value, char category);
	tree::BestSplit getBestSplit();
	size_t totalNumRecordsPassed = 0;
	size_t recordsPassed[NUM_CATEGORIES];
	short valueOfBestSplit = -1;
	bool calculate = true;
	tree::BestSplit bestSplit, overallBestSplit;
	Metadata* metadata;
};

class Metadata {
public:
	static int counter;

	Metadata();
	void increment(char category);
	std::string toString();
	void clean();
	int count;
	size_t totalNumRecords;
	size_t recordsPerCategory[NUM_CATEGORIES];
	float uncertainty;
	tree::BestSplit bestSplit;
	FeatureStats stats[tree::BPT_NUM_FEATURES];
	tree::Node** nodeRef = NULL;
	bool done = false;
};

size_t get_subset(size_t val);
char get_category(size_t val);
bool get_flag(size_t val);
size_t combine(size_t subset, bool flag, char value);

void impurity(Metadata *metadata);
float impurity(FeatureStats* stats);
float impurity(FeatureStats* stats, Metadata* metadata);
float infoGain(FeatureStats* stats, Metadata* metadata);
void bestSplit(Metadata *metadata);

bool isDoneOrNull(Metadata*& m);