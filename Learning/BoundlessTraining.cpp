#include "BoundlessTraining.h"
#include "TreeConstants.h"
#include "TreeSettings.h"
#include <fstream>
#include "TreeUtils.h"
#include <iostream>
#include <sstream>
#include <functional>

using std::string;
using std::to_string;

#define BITS_VALUE 8
#define MASK_VALUE UCHAR_MAX
#define MASK_SUBSET (ULLONG_MAX - UCHAR_MAX)
#define DONE 37

void openFile(FILE* file, string path);
void determineBufferSizes(FILE* file, int* buffer_size, int* line_size);
void iterate(FILE* file, std::vector<Metadata*> metadata, char* buffer, int buffer_size, char* line_buffer, int line_size,
	PlainRecord* recordTemplate, size_t* lookupTable, std::function<void(PlainRecord*, size_t*, std::vector<Metadata*>)> func);

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

class FeatureStats {
public:
	FeatureStats() {
		for (int i = 0; i < NUM_CATEGORIES; i++)
			recordsPassed[i] = 0;
	}
	void nextValue(short value, char category) {
		if (value != lastHandledValue)
			calculate = true;
		lastHandledValue = value;
		totalNumRecordsPassed++;
		recordsPassed[category]++;
	}

	size_t totalNumRecordsPassed = 0;
	size_t recordsPassed[NUM_CATEGORIES];
	short lastHandledValue = -1;
	bool calculate = false;
};

class Metadata {
public:
	Metadata() {
		trace("Create Metadata");
		for (int i = 0; i < tree::BPT_NUM_FEATURES; i++) {
			stats[i] = FeatureStats();
		}
	}
	void increment(char category) {
		//trace("Increment category " + to_string(category));
		totalNumRecords++;
		recordsPerCategory[category]++;
		//trace("New totalNumRecords = " + to_string(totalNumRecords));
		//trace("New records counted for category = " + to_string(recordsPerCategory[category]));
	}
	string toString() {
		std::stringstream ss;
		ss << "Metadata: " << std::endl;
		ss << "totalNumRecords = " << to_string(totalNumRecords) << std::endl;
		for (int c = 0; c < NUM_CATEGORIES; c++) {
			ss << "Category(" << to_string(c) << ") = " << to_string(recordsPerCategory[c]) << std::endl;
		}
		ss << "Uncertainty = " << uncertainty << std::endl;
		ss << "BestSplit = >" << to_string(bestSplit.decision.feature) << "," << to_string(bestSplit.decision.refVal) << "<" << std::endl;
		ss << std::endl;
		
		return ss.str();
	}
	void clean() {
		totalNumRecords = 0;
		for (int i = 0; i < tree::BPT_NUM_FEATURES; i++) {
			stats[i] = FeatureStats();
		}
		for (int c = 0; c < NUM_CATEGORIES; c++) {
			recordsPerCategory[c] = 0;
		}
		bestSplit.gain = 0;
		uncertainty = 0;
	}
	size_t totalNumRecords;
	size_t recordsPerCategory[NUM_CATEGORIES];
	float uncertainty;
	tree::BestSplit bestSplit;
	FeatureStats stats[tree::BPT_NUM_FEATURES];
	tree::Node* node;
	bool done;
};

size_t get_subset(size_t val) {
	//trace(to_string(val) + " has subset " + to_string((val & MASK_SUBSET) >> BITS_VALUE));

	//std::stringstream ss;
	//ss << "Value: ";
	//ss << std::showbase << std::uppercase << std::setfill('0') << std::setw(4) << std::hex << val;
	//ss << std::endl;
	//ss << "Mask: ";
	//ss << std::showbase << std::uppercase << std::setfill('0') << std::setw(4) << std::hex << MASK_SUBSET;
	//ss << std::endl;
	//trace(ss.str());

	return (val & MASK_SUBSET) >> BITS_VALUE;
}

char get_category(size_t val) {
	//trace(to_string(val) + " has category " + to_string(val & MASK_VALUE));
	//std::stringstream ss;
	//ss << "Value: ";
	//ss << std::showbase << std::uppercase << std::setfill('0') << std::setw(4) << std::hex << val;
	//ss << std::endl;
	//ss << "Mask: ";
	//ss << std::showbase << std::uppercase << std::setfill('0') << std::setw(4) << std::hex << MASK_VALUE;
	//ss << std::endl;
	//trace(ss.str());
	return (char)(val & MASK_VALUE);
}

size_t combine(size_t subset, char value) {
	trace(to_string(subset) + " and " + to_string(value) + " combined is " + to_string(((subset << BITS_VALUE) & MASK_SUBSET) | (value & MASK_VALUE)));
	return ((subset << BITS_VALUE) & MASK_SUBSET) | (value & MASK_VALUE);
}

void impurity(Metadata *metadata) {
	trace("impurity(Metadata)");
	double uncertainty = 1;
	size_t totalNumber = metadata->totalNumRecords;
	//trace("TotalNumber = " + to_string(totalNumber));

	for (short i = 0; i < NUM_CATEGORIES; i++) {
		size_t number = metadata->recordsPerCategory[i];
		double reduce = pow(((double)number / (double)totalNumber), 2);
		//trace("Reduce = " + to_string(reduce));
		uncertainty -= reduce;
		//trace("Decreased uncertainty = " + to_string(uncertainty));
	}

	metadata->uncertainty = (float)uncertainty;
	trace("uncertainty = " + to_string(metadata->uncertainty));
}

float impurity(FeatureStats* stats) {
	trace("impurity(Featurestats)");
	double uncertainty = 1;
	size_t totalNumber = stats->totalNumRecordsPassed;
	trace("TotalNumber = " + to_string(totalNumber));

	for (short i = 0; i < NUM_CATEGORIES; i++) {
		size_t number = stats->recordsPassed[i];
		double reduce = pow(((double)number / (double)totalNumber), 2);
		trace("Reduce = " + to_string(reduce));
		uncertainty -= reduce;
		trace("Decreased uncertainty = " + to_string(uncertainty));
	}

	trace("uncertainty = " + to_string(uncertainty));

	return (float)uncertainty;
}

float impurity(FeatureStats* stats, Metadata* metadata) {
	trace("impurity(FeatureStats, Metadata)");
	double uncertainty = 1;
	size_t totalNumber = metadata->totalNumRecords - stats->totalNumRecordsPassed;
	trace("TotalNumber = " + to_string(totalNumber));

	for (short i = 0; i < NUM_CATEGORIES; i++) {
		size_t number = metadata->recordsPerCategory[i] - stats->recordsPassed[i];
		double reduce = pow(((double)number / (double)totalNumber), 2);
		trace("Reduce = " + to_string(reduce));
		uncertainty -= reduce;

		trace("Decreased uncertainty = " + to_string(uncertainty));
	}

	trace("uncertainty = " + to_string(uncertainty));

	return (float)uncertainty;
}

float infoGain(FeatureStats* stats, Metadata* metadata) {
	trace("infoGain()");
	float p = (float) stats->totalNumRecordsPassed / (float) metadata->totalNumRecords;
	float infoGain = metadata->uncertainty - p * impurity(stats) - (1 - p) * impurity(stats, metadata);
	trace("infoGain = " + to_string(infoGain));
	return infoGain;
}

void bestSplit(Metadata *metadata) {
	trace("bestSplit()");
	float bestInfoGain = metadata->bestSplit.gain;
	float tempInfoGain;
	tree::Decision dec;

	for (int f = 0; f < tree::BPT_NUM_FEATURES; f++) {
		if (metadata->stats[f].calculate) {
			trace("Calculate feature <" + to_string(f) + ">");
			tempInfoGain = infoGain(&metadata->stats[f], metadata);
			if (tempInfoGain > bestInfoGain) {
				bestInfoGain = tempInfoGain;
				dec.feature = f;
				dec.refVal = metadata->stats[f].lastHandledValue;
				trace("tempInfoGain > bestInfoGain -> set Decision to <" + to_string(dec.feature) + "," + to_string(dec.refVal) + ">");
			}
			metadata->stats[f].calculate = false;
		}
	}

	if (bestInfoGain > metadata->bestSplit.gain) {
		metadata->bestSplit.gain = bestInfoGain;
		metadata->bestSplit.decision = dec;

		trace("bestInfoGain > bestSplit.gain -> new BestSplit = <" + to_string(metadata->bestSplit.gain) + "|" + to_string(metadata->bestSplit.decision.feature)
			+ "," + to_string(metadata->bestSplit.decision.refVal) + ">");
	}
}

void readMetaData(Metadata* metadata, size_t* lookupTable) {
	trace("readMetaData()");
	// when Format of Metadata is known -> include reading from file
	// for now set manually
	metadata->totalNumRecords = tree::BPT_NUM_RECORDS;
	
	for (size_t i = 0; i < tree::BPT_NUM_RECORDS; i++) {
		metadata->recordsPerCategory[get_category(lookupTable[i])]++;
	}

	for (int i = 0; i < NUM_CATEGORIES; i++) {
		trace("Metadata-Category(" + to_string(i) + ") = " + to_string(metadata->recordsPerCategory[i]));
	}

	//metadata->recordsPerCategory[0] = tree::BPT_NUM_CATEGORY_1;
	//metadata->recordsPerCategory[1] = tree::BPT_NUM_CATEGORY_2;
	//metadata->recordsPerCategory[2] = tree::BPT_NUM_CATEGORY_3;
	//metadata->recordsPerCategory[3] = tree::BPT_NUM_CATEGORY_4;
	//metadata->recordsPerCategory[4] = tree::BPT_NUM_CATEGORY_5;
	//metadata->recordsPerCategory[5] = tree::BPT_NUM_CATEGORY_6;

	impurity(metadata);
	trace("readMetaData() finished");
}

void loadLookupTable(size_t* lookupTable, string datasetMap) {
	trace("loadLookupTable()");
	FILE* map;
	fopen_s(&map, datasetMap.c_str(), "r");

	int line_size = sizeof(MapRecord);
	int buffer_size = line_size * tree::BPT_NUM_RECORDS_IN_BUFFER;
	char* buffer = new char[buffer_size];
	char* line_buffer = new char[line_size];

	MapRecord* record;
	record = (MapRecord*)&line_buffer[0];

	size_t read_bytes, index;
	short value;

	while (read_bytes = fread(buffer, 1, buffer_size, map)) {
		if (read_bytes == (size_t)-1) {
			printf("Read failed!\n");
			string c;
			std::cin >> c;
			return exit(5);
		}

		if (!read_bytes)
			break;

		for (int i = 0; read_bytes > 0; i++) {
			memcpy(line_buffer, &buffer[i*line_size], line_size);
			//line_buffer[line_size - 1] = '\0';
			read_bytes -= line_size;

			index = atoi(record->record);
			value = atoi(&record->category);

			lookupTable[index] = value;
			//trace("Map(" + to_string(index) + ") = " + to_string(value));
		}
	}

	delete[] line_buffer;
	delete[] buffer;

	fclose(map);
	trace("Lookup Table built.");
}

void handleRecord(PlainRecord* record, size_t* lookupTable, std::vector<Metadata>& metadatas, bool checkSubset = false) {
	trace("handleRecord()");
	RecordValuePair rVPair;
	size_t rec;
	size_t subset = 0;
	short value;
	short category;

	for (int feat = 0; feat < tree::BPT_NUM_FEATURES; feat++) {
		rVPair = record->feature[feat];
		rec = atoi(rVPair.record);
		value = atoi(rVPair.value);
		category = get_category(lookupTable[rec]);
		if (checkSubset) {
			subset = get_subset(lookupTable[rec]);
		}
		
		metadatas[subset].stats[feat].nextValue(value, category);
	}

	for (Metadata& meta : metadatas) {
		bestSplit(&meta);
	}
}

void training(FILE* file, std::vector<Metadata>& metadatas, size_t* lookupTable, PlainRecord* recordTemplate, int buffer_size, int line_size) {

	trace("training()");
	std::vector<Metadata> tempMetadata;
	char* buffer = new char[buffer_size];
	char* line_buffer = new char[line_size];

	bool finished = false;
	std::vector<short> checkFeatures;

	while (!finished) {

		finished = true;
		fseek(file, 0, SEEK_SET);

		size_t read_bytes, subset;
		while (read_bytes = fread(buffer, 1, buffer_size, file)) {
			if (read_bytes == (size_t)-1) {
				printf("Read failed!\n");
				return exit(2);
			}

			if (!read_bytes)
				break;

			for (int i = 0; read_bytes > 0; i++) {
				memcpy(line_buffer, &buffer[i*line_size], line_size);
				line_buffer[line_size - 1] = '\0';
				read_bytes -= line_size;

				handleRecord(recordTemplate, lookupTable, metadatas, true);
			}
		}

		//add nodes to metadata and connect it properly!

		trace("add features to check");
		checkFeatures.clear();
		for (std::vector<Metadata>::iterator m = metadatas.begin(); m != metadatas.end(); ++m) {
			if (std::find(checkFeatures.begin(), checkFeatures.end(), m->bestSplit.decision.feature) == checkFeatures.end()) {
				checkFeatures.push_back(m->bestSplit.decision.feature);
			}
			m->clean();
		}

		size_t size = metadatas.size();
		trace("metadatas size = " + to_string(size));
		for (int i = 1; i < size; i += 2) {
			metadatas.insert(metadatas.begin() + i, Metadata());
		}
		metadatas.insert(metadatas.end(), Metadata());
		trace("metadatas size = " + to_string(metadatas.size()));

		for (Metadata m : metadatas) {
			trace(m.toString());
		}

		fseek(file, 0, SEEK_SET); // goto beginning of File

		size_t record, look;
		char value, category;
		while (read_bytes = fread(buffer, 1, buffer_size, file)) {
			if (read_bytes == (size_t)-1) {
				printf("Read failed!\n");
				return exit(4);
			}

			if (!read_bytes)
				break;

			for (int i = 0; read_bytes > 0; i++) {
				memcpy(line_buffer, &buffer[i*line_size], line_size);
				read_bytes -= line_size;

				for (short feat : checkFeatures) {
					record = atoi(recordTemplate->feature[feat].record);
					value = atoi(recordTemplate->feature[feat].value);
					look = lookupTable[record];
					category = get_category(look);
					subset = get_subset(look);
					subset *= 2;

					if (value < metadatas[subset].bestSplit.decision.refVal) {
						metadatas[subset].increment(category);
					}
					else {
						subset += 1;
						metadatas[subset].increment(category);
					}

					lookupTable[record] = combine(subset, category);
				}
			}
		}
	}

	delete[] buffer;
	delete[] line_buffer;
	trace("training finished");
}

void startBoundlessTraining(string datasetFile, string datasetMap, tree::Node*& rootNode) {

	size_t* lookupTable = new size_t[tree::BPT_NUM_RECORDS];
	loadLookupTable(lookupTable, datasetMap);

	Metadata metadata;
	readMetaData(&metadata, lookupTable);

	FILE *file;
	openFile(file, datasetFile);

	int buffer_size, line_size;
	determineBufferSizes(file, &buffer_size, &line_size);

	printf("Create buffer with size %d.\n", buffer_size);
	char* buffer = new char[sizeof(char) * buffer_size]; //malloc(sizeof(char) * buffer_size);
	char* line_buffer = new char[sizeof(char) * line_size]; //malloc(sizeof(char) * line_size);

	PlainRecord* recordTemplate;
	recordTemplate = (PlainRecord*)&line_buffer[0];

	std::vector<Metadata> meta;
	meta.push_back(metadata);

	iterate(file, std::vector<Metadata*>(), buffer, buffer_size, line_buffer, line_size, recordTemplate, lookupTable, NULL);

	size_t read_bytes;
	while (read_bytes = fread(buffer, 1, buffer_size, file)) {
		if(read_bytes == (size_t)-1) {
			printf("Read failed!\n");
			return exit(2);
		}

		if(!read_bytes)
			break;

		for(int i = 0; read_bytes > 0; i++) {
			memcpy(line_buffer, &buffer[i*line_size], line_size);
			line_buffer[line_size - 1] = '\0';
			read_bytes -= line_size;

			handleRecord(recordTemplate, lookupTable, meta);
		}
	}

	rootNode = (tree::Node*) new tree::DecisionNode(metadata.bestSplit.decision);
	metadata.node = rootNode;

	fseek(file, 0, SEEK_SET); // goto beginning of File

	short feat = metadata.bestSplit.decision.feature;
	short refVal = metadata.bestSplit.decision.refVal;
	size_t look, record, subset;
	char category;
	short value;
	std::vector<Metadata> metadatas;
	metadatas.insert(metadatas.begin(), Metadata());
	metadatas.insert(metadatas.begin()+1, Metadata());
	metadatas[0].node = rootNode->true_branch;
	metadatas[1].node = rootNode->false_branch;

	while (read_bytes = fread(buffer, 1, buffer_size, file)) {
		if (read_bytes == (size_t)-1) {
			printf("Read failed!\n");
			return exit(4);
		}

		if (!read_bytes)
			break;

		for (int i = 0; read_bytes > 0; i++) {
			memcpy(line_buffer, &buffer[i*line_size], line_size);
			//line_buffer[line_size - 1] = '\0';
			read_bytes -= line_size;

			record = atoi(recordTemplate->feature[feat].record);
			value = atoi(recordTemplate->feature[feat].value);
			look = lookupTable[record];
			category = get_category(look);
			subset = get_subset(look);

			if (value < refVal) {
				metadatas[0].increment(category);
				subset *= 2;
			}
			else {
				metadatas[1].increment(category);
				subset = subset * 2 + 1;
			}

			lookupTable[record] = combine(subset, category);
		}
	}

	impurity(&metadatas[0]);
	impurity(&metadatas[1]);

	//trace("Metadata:");
	//trace(metadata.toString());
	//trace("Metadata1:");
	//trace(metadatas[0].toString());
	//trace("Metadata2:");
	//trace(metadatas[1].toString());
	printf("Close Main buffers.\n");
	delete buffer;
	delete line_buffer;

	training(file, metadatas, lookupTable, recordTemplate, buffer_size, line_size);

	delete[] lookupTable;

	printf("Close %s.\n", datasetFile.c_str());
	fclose(file);
}

void openFile(FILE* file, string path) {
	printf("Open %s.\n", path.c_str());
	fopen_s(&file, path.c_str(), "r");

	if (file == NULL) {
		printf("File couldn't be opened! Exit.\n");
		string s;
		std::cin >> s;
		exit(1);
	}
}

void determineBufferSizes(FILE* file, int* buffer_size, int* line_size) {
	*buffer_size = 1; //last char won't be counter (\n) -> start with 1
	while (fgetc(file) != '\n') {
		*buffer_size++;
	}

	if (sizeof(PlainRecord) != *buffer_size) {
		printf("Record Template doesn't fit!\n");
		string a;
		std::cin >> a;
		exit(7);
	}

	*line_size = *buffer_size;
	*buffer_size *= tree::BPT_NUM_RECORDS_IN_BUFFER;
}

void iterate(FILE* file, std::vector<Metadata*> metadata, char* buffer, int buffer_size, char* line_buffer, int line_size, 
	PlainRecord* recordTemplate, size_t* lookupTable, std::function<void(PlainRecord*,size_t*,std::vector<Metadata*>)> func) {
	
	fseek(file, 0, SEEK_SET); // goto beginning of File
	
	size_t read_bytes;
	while (read_bytes = fread(buffer, 1, buffer_size, file)) {
		if (read_bytes == (size_t)-1) {
			printf("Read failed!\n");
			return exit(2);
		}

		if (!read_bytes)
			break;

		for (int i = 0; read_bytes > 0; i++) {
			memcpy(line_buffer, &buffer[i*line_size], line_size);
			read_bytes -= line_size;

			func(recordTemplate, lookupTable, metadata);
			//handleRecord(recordTemplate, lookupTable, meta);
		}
	}
}