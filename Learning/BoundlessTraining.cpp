#include "BoundlessTraining.h"
#include "BoundlessTrainingUtils.h"
#include "TreeConstants.h"
#include "TreeSettings.h"
#include <fstream>
#include "TreeUtils.h"
#include <iostream>
#include <sstream>
#include <functional>
#include <ctime>
#include <map>

#define REC_DONE ULLONG_MAX

using std::string;
using std::to_string;

void openFile(FILE*& file, string path);
void determineBufferSizes(FILE*& file, int* buffer_size, int* line_size);
void iterate(FILE* file, std::vector<Metadata*>& metadata, std::vector<short>& features, char* buffer, int buffer_size, char* line_buffer, int line_size,
	PlainRecord* recordTemplate, size_t* lookupTable, bool assign, int iteration, 
	std::function<void(PlainRecord*, size_t*, std::vector<Metadata*>&, std::vector<short>& features, bool b)> func);
bool processMetadata(std::vector<Metadata*>& metadatas, std::vector<short>& checkFeatures, size_t* lookupTable);
bool postProcess(std::vector<Metadata*>& metadatas);
void printLookupTable(size_t* lookupTable);

void readMetaData(Metadata* metadata, size_t* lookupTable) {
	// when Format of Metadata is known -> include reading from file
	// for now set manually
	metadata->totalNumRecords = tree::BPT_NUM_RECORDS;
	
	for (size_t i = 0; i < tree::BPT_NUM_RECORDS; i++) {
		metadata->recordsPerCategory[get_category(lookupTable[i])]++;
	}

	//metadata->recordsPerCategory[0] = tree::BPT_NUM_CATEGORY_1;
	//metadata->recordsPerCategory[1] = tree::BPT_NUM_CATEGORY_2;
	//metadata->recordsPerCategory[2] = tree::BPT_NUM_CATEGORY_3;
	//metadata->recordsPerCategory[3] = tree::BPT_NUM_CATEGORY_4;
	//metadata->recordsPerCategory[4] = tree::BPT_NUM_CATEGORY_5;
	//metadata->recordsPerCategory[5] = tree::BPT_NUM_CATEGORY_6;

	impurity(metadata);
}

void loadLookupTable(size_t* lookupTable, string datasetMap) {
	FILE* map;
	fopen_s(&map, datasetMap.c_str(), "r");

	if (map == NULL) {
		printf("%s could not be opened! -> exit.\n", datasetMap.c_str());
		exit(7);
	}

	int line_size = sizeof(MapRecord);
	int buffer_size = line_size * tree::BPT_NUM_RECORDS_IN_BUFFER;
	char* buffer = new char[buffer_size];
	char* line_buffer = new char[line_size];

	MapRecord* record;
	record = (MapRecord*)&line_buffer[0];

	size_t read_bytes, index, num_records = 0;
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
			read_bytes -= line_size;

			index = atol(record->record);
			value = atoi(&record->category);

			lookupTable[index] = value;
			num_records++;
		}
	}

	delete[] line_buffer;
	delete[] buffer;

	if (map != NULL) {
		printf("Closing File %s\n", datasetMap.c_str());
		fclose(map);
	}

	printf("Num Records read: %lld\n", num_records);

	if (num_records != tree::BPT_NUM_RECORDS) {
		trace("Setting tree::BPT_NUM_RECORDS doesn't match the number of records!");
		printf("Num_Records(%lld) != BPT_NUM_RECORDS(%d)\n", num_records, tree::BPT_NUM_RECORDS);
		exit(70);
	}
}

void checkRecord(PlainRecord* record, size_t* lookupTable, std::vector<Metadata*>& metadatas, std::vector<short>& features, bool checkSubset) {
	RecordValuePair rVPair;
	size_t rec, look;
	size_t subset = 0;
	short value;
	char category;

	for (int feat = 0; feat < tree::BPT_NUM_FEATURES; feat++) {
		rVPair = record->feature[feat];
		rec = atol(rVPair.record);

		look = lookupTable[rec];
		
		if (look == REC_DONE)
			continue;

		value = atoi(rVPair.value);
		category = get_category(look);
		if (checkSubset) {
			subset = get_subset(look);
		}

		if (subset >= metadatas.size()) {
			trace("subset is too big!!! - num Metadatas = " + to_string(metadatas.size()) + ", subset = " + to_string(subset));
			printLookupTable(lookupTable);
		}

		if (isDoneOrNull(metadatas[subset])) {
			continue;
		}

		metadatas[subset]->stats[feat].nextValue(value, category);
	}
}

void assignRecord(PlainRecord* recordTemplate, size_t* lookupTable, std::vector<Metadata*>& metadatas, std::vector<short>& features, bool processFlag) {
	size_t look, record, subset;
	char category;
	int value;
	//short feat;
	bool processed;
	
	//size_t size = metadatas.size();
	//for (int i = 0; i < size; i += 2) {
	//	if (isDoneOrNull(metadatas[i]))
	//		continue;
	for(short feat : features) {

		//feat = metadatas[i]->bestSplit.decision.feature;

		//if (std::find(features.begin(), features.end(), feat) == features.end()) {
		//	trace("Error with checkFeatures!");
		//}

		record = atol(recordTemplate->feature[feat].record);
		look = lookupTable[record];

		if (look == REC_DONE)
			continue;

		processed = get_flag(look);
		if (processed != processFlag) {
			continue;
		}

		subset = get_subset(look);

		if (subset * 2 >= metadatas.size()) {
			trace("subset is too big!!! - num Metadatas == " + to_string(metadatas.size()) + ", check feature " + to_string(feat) + 
				", subset = " + to_string(subset * 2) + ", Record checked is " + to_string(record));
			printLookupTable(lookupTable);
			int a = 1 + 2;
		}

		if (isDoneOrNull(metadatas[subset*2]) || metadatas[subset*2]->bestSplit.decision.feature != feat)
			continue;

		category = get_category(look);
		value = atoi(recordTemplate->feature[feat].value);
		
		subset *= 2;

		if (value <= metadatas[subset]->bestSplit.decision.refVal) {
			metadatas[subset]->increment(category);
		}
		else {
			subset += 1;
			metadatas[subset]->increment(category);
		}
			
		lookupTable[record] = combine(subset, !processed, category);
	}
}

void printMetadata(std::vector<Metadata*>& metadatas) {
	trace("Metadatas: ");
	trace("size = " + to_string(metadatas.size()));

	for (int i = 0; i < metadatas.size(); i++) {
		if (metadatas[i] == NULL) {
			trace("Metadatas(" + to_string(i) + ") == NULL");
		}
		else {
			trace("Metadatas(" + to_string(i) + ") == NOT NULL");
			trace(metadatas[i]->toString());
		}
	}
}

//void printRecords(size_t* lookupTable, size_t pSubset) {
//	size_t look, subset;
//	
//	trace("Subset " + to_string(pSubset) + ":");
//
//	//for (size_t i = 0; i < tree::BPT_NUM_RECORDS; i++) {
//	//	look = lookupTable[i];
//	//	subset = get_subset(look);
//
//	//	if (subset == pSubset) {
//	//		std::stringstream ss;
//	//		ss << "Record(" << std::setw(2) << std::setfill('0') << i << ") = 0x" << std::setw(4) << std::setfill('0') << std::hex << look;
//	//		trace(ss.str());
//	//	}
//	//}
//}

void printLookupTable(size_t* lookupTable) {
	size_t look, subset;

	trace("Lookup Table:");

	for (size_t i = 0; i < tree::BPT_NUM_RECORDS; i++) {
		look = lookupTable[i];
		subset = get_subset(look);
		std::stringstream ss;
		//ss << "Record(" << std::setw(4) << std::setfill('0') << i << ") = 0x" << std::setw(14) << std::setfill('0') << std::hex << subset;
		ss << "Record(" << i << ") = " << std::setw(12) << std::setfill('0') << subset;
		trace(ss.str());
	}
}

void startBoundlessTraining(string datasetFile, string datasetMap, tree::Node** rootNode) {

	size_t* lookupTable = new size_t[tree::BPT_NUM_RECORDS];
	loadLookupTable(lookupTable, datasetMap);

	std::vector<Metadata*> metadatas;
	metadatas.push_back(new Metadata());
	metadatas[0]->nodeRef = rootNode;
	readMetaData(metadatas[0], lookupTable);

	std::vector<short> checkFeatures;

	FILE *file = NULL;
	openFile(file, datasetFile);

	int buffer_size = 0, line_size = 0;
	determineBufferSizes(file, &buffer_size, &line_size);

	char* buffer = new char[sizeof(char) * buffer_size]; //malloc(sizeof(char) * buffer_size);
	char* line_buffer = new char[sizeof(char) * line_size]; //malloc(sizeof(char) * line_size);

	PlainRecord* recordTemplate;
	recordTemplate = (PlainRecord*)&line_buffer[0];

	time_t start, end;
	double elapsed;
	int iteration = 0;
	bool processing = true;
	while (processing) {
		trace("Iteration " + to_string(iteration++));
		printf("Iteration %d\n", iteration - 1);

		start = clock();
		try {
			iterate(file, metadatas, checkFeatures, buffer, buffer_size, line_buffer, line_size, recordTemplate, 
				lookupTable, false, iteration, checkRecord);
		}
		catch (...) {
			trace("Exception in First Iteration!");
			trace("");
			printMetadata(metadatas);
			trace("");
			printLookupTable(lookupTable);
			exit(98);
		}
		end = clock();
		elapsed = ((double)end - (double)start) / (double)CLOCKS_PER_SEC;
		printf("Iteration 1 took %05.2lf seconds.\n", elapsed);
		trace("Iteration 1 took " + to_string(elapsed) + " seconds.");

		start = clock();
		try {
			processing = processMetadata(metadatas, checkFeatures, lookupTable);
		}
		catch (...) {
			trace("Exception in processMetadata!");
			trace("");
			printMetadata(metadatas);
			trace("");
			printLookupTable(lookupTable);
			exit(99);
		}

		end = clock();
		elapsed = ((double)end - (double)start) / (double)CLOCKS_PER_SEC;
		printf("processMetadata took %05.2lf seconds.\n", elapsed);
		trace("processMetadata took " + to_string(elapsed) + " seconds.");

		if (!processing) {
			trace("processing = false -> break");
			break;
		}

		trace("Num Features to check: " + to_string(checkFeatures.size()));

		start = clock();
		try {
			iterate(file, metadatas, checkFeatures, buffer, buffer_size, line_buffer, line_size, recordTemplate, lookupTable, true, iteration, assignRecord);
		}
		catch (...) {
			trace("Exception in Second Iteration!");
			trace("");
			printMetadata(metadatas);
			trace("");
			printLookupTable(lookupTable);
		}

		end = clock();
		elapsed = ((double)end - (double)start) / (double)CLOCKS_PER_SEC;
		printf("Iteration 2 took %05.2lf seconds.\n", elapsed);
		trace("Iteration 2 took " + to_string(elapsed) + " seconds.");

		start = clock();
		try {
			processing = postProcess(metadatas);
		}
		catch (...) {
			trace("Exception in postProcess!");
			trace("");
			printMetadata(metadatas);
			trace("");
			printLookupTable(lookupTable);
		}

		end = clock();
		elapsed = ((double)end - (double)start) / (double)CLOCKS_PER_SEC;
		printf("postProcess took %05.2lf seconds.\n", elapsed);
		trace("postProcess took " + to_string(elapsed) + " seconds.");
	}

	trace("Finished loop");

	printf("Close Main buffers.\n");
	delete[] buffer;
	delete[] line_buffer;

	for (Metadata*& m : metadatas) {
		if (m != NULL) {
			delete m;
			m = NULL;
			trace("Delete Metadata");
		}
	}

	delete[] lookupTable;

	printf("Close %s.\n", datasetFile.c_str());
	fclose(file);
}

void openFile(FILE*& file, string path) {
	printf("Open %s.\n", path.c_str());
	fopen_s(&file, path.c_str(), "r");

	if (file == NULL) {
		printf("File couldn't be opened! Exit.\n");
		string s;
		std::cin >> s;
		exit(1);
	}
}

void determineBufferSizes(FILE*& file, int* buffer_size, int* line_size) {
	*buffer_size = 1; //last char won't be counter (\n) -> start with 1
	while (fgetc(file) != '\n') {
		(*buffer_size)++;
	}

	printf("Buffer_size: %d\n", *buffer_size);

	if (sizeof(PlainRecord) != *buffer_size) {
		printf("Record Template doesn't fit!\n");
		string a;
		std::cin >> a;
		exit(7);
	}

	*line_size = *buffer_size;
	*buffer_size = *buffer_size * tree::BPT_NUM_RECORDS_IN_BUFFER;
}

void iterate(FILE* file, std::vector<Metadata*>& metadata, std::vector<short>& features, char* buffer, int buffer_size, char* line_buffer, 
	int line_size, PlainRecord* recordTemplate, size_t* lookupTable, bool assign, int iteration,
	std::function<void(PlainRecord*,size_t*,std::vector<Metadata*>&, std::vector<short>& features, bool b)> func) {
	
	bool flag = metadata.size() > 1;

	if (assign) {
		flag = iteration % 2 == 0;
	}

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

			func(recordTemplate, lookupTable, metadata, features, flag);
		}
	}
}

bool processMetadata(std::vector<Metadata*>& metadatas, std::vector<short>& checkFeatures, size_t* lookupTable) {

	trace("processMetadata()");
	std::map<size_t, size_t> subsetChangeMap;
	std::vector<size_t> subsetCrossOutList;

	bool done = true;
	short feature;

	checkFeatures.clear();

	for (Metadata*& m : metadatas) {
		if (isDoneOrNull(m)) {
			continue;
		}

		bestSplit(m);

		feature = m->bestSplit.decision.feature;
		if (checkFeatures.size() != tree::BPT_NUM_FEATURES) {
			if (std::find(checkFeatures.begin(), checkFeatures.end(), feature) == checkFeatures.end()) {
				checkFeatures.push_back(feature);
				trace("Add Feature " + to_string(feature) + " to checkFEatures");
			}
		}

		done = false;

		if (m->nodeRef == NULL) {
			trace("Metadata " + to_string(m->count) + ") has no nodeRef!");
			printf("NodeRef is NULL! ERROR.\n");
			printf("Check the insertion mechanism for issues!\n");
		}

		*m->nodeRef = (tree::Node*) new tree::DecisionNode(m->bestSplit.decision);
		if (*m->nodeRef == NULL) {
			printf("Assigning DecisionNode failed\n");
			trace("Assigning DecisionNode failed!");
			exit(31);
		}

		//trace("DecisionNode: >" + (*m->nodeRef)->toString() + "<");
		m->clean();
	}

	if (done) {
		trace("Job DONE -> return");
		return false;
	}

	size_t size = metadatas.size();
	size_t size_before_reduction = size;
	trace("Num Metadatas = " + to_string(size));

	std::vector<Metadata*>::iterator begin = metadatas.begin(), end = metadatas.end() - 1;
	size_t index_begin = 0, index_end = 0;

	while (begin != end) {
		if (*begin != NULL) {
			++begin;
		}
		else if (*end == NULL) {
			//trace("Delete end");
			index_end = end - metadatas.begin();
			subsetCrossOutList.push_back(index_end);
			metadatas.erase(end--);
		}
		else if (*end != NULL) {
			index_begin = begin - metadatas.begin();
			index_end = end - metadatas.begin();
			subsetChangeMap.insert(std::pair<size_t, size_t>(index_end, index_begin));
			subsetCrossOutList.push_back(index_begin);
			//trace("SEt index(" + to_string(index_begin) + ") to value of index(" + to_string(index_end) + ") and remove end!");
			metadatas[index_begin] = metadatas[index_end];
			metadatas.erase(end--);
		}
	}

	if (begin == end && *end == NULL) {
		//trace("Delete end");
		index_end = end - metadatas.begin();
		subsetCrossOutList.push_back(index_end);
		metadatas.erase(end);
	}

	size = metadatas.size();
	trace("After Reduction size = " + to_string(size));

	if (size > 300 && size == size_before_reduction) {
		trace("Metadata size does not reduce -> ERROR?!");
		printMetadata(metadatas);
		printLookupTable(lookupTable);
		exit(17);
	}

	for (int i = 1; i < size; i++) {
		metadatas.insert(metadatas.begin() + (i * 2 - 1), new Metadata());
		metadatas[i * 2 - 1]->bestSplit = metadatas[(i - 1) * 2]->bestSplit;
	}

	metadatas.insert(metadatas.end(), new Metadata());
	size = metadatas.size();
	metadatas[size - 1]->bestSplit = metadatas[size - 2]->bestSplit;

	trace("Num Metadatas after Insertion = " + to_string(size));

	if (size % 2 != 0) {
		printf("Error while processing metadata. size is odd!\n");
		exit(19);
	}

	size = metadatas.size();
	for (int i = 0; i < size; i += 2) {
		if (isDoneOrNull(metadatas[i]))
			continue;

		metadatas[i + 1]->nodeRef = &(*metadatas[i]->nodeRef)->false_branch;
		metadatas[i]->nodeRef = &(*metadatas[i]->nodeRef)->true_branch;

		if (metadatas[i]->nodeRef == NULL) {
			trace("Assigning False Branch failed!");
			exit(71);
		}

		if (metadatas[i + 1]->nodeRef == NULL) {
			trace("Assigning True Branch failed!");
			exit(73);
		}
	}

	if (!subsetCrossOutList.empty()) {
		size_t look, subset, temp;
		char value;
		bool flag;
		std::map<size_t, size_t>::iterator element;

		for (size_t i = 0; i < tree::BPT_NUM_RECORDS; i++) {
			look = lookupTable[i];

			if (look == REC_DONE)
				continue;

			subset = get_subset(look);

			if ((element = subsetChangeMap.find(subset)) != subsetChangeMap.end()) {
				value = get_category(look);
				flag = get_flag(look);
				temp = combine(element->second, flag, value);
				lookupTable[i] = temp;
			}
			else if (std::find(subsetCrossOutList.begin(), subsetCrossOutList.end(), subset) != subsetCrossOutList.end()) {
				lookupTable[i] = REC_DONE;
			}
		}
	}

	printf("Num Metadatas: %lld\n", metadatas.size());
	trace("processMetadata() done");

	return !done;
}

std::vector<tree::Result> as_result(Metadata* metadata) {
	std::vector<tree::Result> results;

	size_t total = metadata->totalNumRecords, category;
	for (int i = 0; i < NUM_CATEGORIES; i++) {
		category = metadata->recordsPerCategory[i];

		if (category > 0) {
			tree::Result r;
			r.outcome = categoryOfValue(i);
			r.probability = (float) ((double)category / (double)total);
			results.push_back(r);
		}
	}

	return results;
}

bool postProcess(std::vector<Metadata*>& metadatas) {
	
	bool done = true;
	for (Metadata*& m : metadatas) {
		if (isDoneOrNull(m))
			continue;

		done = false;
		impurity(m);

		if (m->uncertainty <= tree::BPT_STOP_EVALUATION_IMPURITY || m->totalNumRecords <= tree::BPT_STOP_EVALUATION_LIMIT) {
			*m->nodeRef = (tree::Node*) new tree::ResultNode(as_result(m));
			//trace("ResultNode: " + (*m->nodeRef)->toString());
			m->done = true;
			delete m;
			m = NULL;
		}
	}

	return !done;
}