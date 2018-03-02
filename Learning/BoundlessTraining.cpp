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

using std::string;
using std::to_string;

void openFile(FILE*& file, string path);
void determineBufferSizes(FILE*& file, int* buffer_size, int* line_size);
void iterate(FILE* file, std::vector<Metadata*>& metadata, char* buffer, int buffer_size, char* line_buffer, int line_size,
	PlainRecord* recordTemplate, size_t* lookupTable, bool assign, int iteration, 
	std::function<void(PlainRecord*, size_t*, std::vector<Metadata*>&, bool b)> func);
bool processMetadata(std::vector<Metadata*>& metadatas);
bool postProcess(std::vector<Metadata*>& metadatas);

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
			read_bytes -= line_size;

			index = atol(record->record);
			value = atoi(&record->category);

			lookupTable[index] = value;
		}
	}

	delete[] line_buffer;
	delete[] buffer;

	if (map != NULL) {
		printf("Closing File %s\n", datasetMap.c_str());
		fclose(map);
	}
}

void checkRecord(PlainRecord* record, size_t* lookupTable, std::vector<Metadata*>& metadatas, bool checkSubset) {
	//trace("checkRecord()");
	RecordValuePair rVPair;
	size_t rec;
	size_t subset = 0;
	short value;
	char category;

	for (int feat = 0; feat < tree::BPT_NUM_FEATURES; feat++) {
		//trace("Feature " + to_string(feat));
		rVPair = record->feature[feat];
		rec = atol(rVPair.record);
		value = atoi(rVPair.value);
		category = get_category(lookupTable[rec]);
		if (checkSubset) {
			subset = get_subset(lookupTable[rec]);
		}

		if (isDoneOrNull(metadatas[subset])) {
			//trace("Metadata done -> continue");
			continue;
		}
		//trace("Subset " + to_string(subset) + ", feature " + to_string(feat) + "record " + to_string(rec) + " -> nextValue(" + to_string(value) + "," + to_string(category));
		metadatas[subset]->stats[feat].nextValue(value, category);
	}

	//int i = -1;
	//for (Metadata* meta : metadatas) {
	//	i++;
	//	if (meta->done)
	//		continue;

	//	trace("Metadata(" + to_string(i) + ") bestSplit:");
	//	trace("Current best Info Gain = " + to_string(meta->bestSplit.gain));
	//	bestSplit(meta);
	//}
}

void assignRecord(PlainRecord* recordTemplate, size_t* lookupTable, std::vector<Metadata*>& metadatas, bool processFlag) {
	//trace("assignRecord()");

	size_t look, record, subset, comb;
	char category;
	int value;
	short feat;
	bool processed;
	
	size_t size = metadatas.size();
	for (int i = 0; i < size; i += 2) {
		if (isDoneOrNull(metadatas[i]))
			continue;

		feat = metadatas[i]->bestSplit.decision.feature;

		//trace("Check FEature >" + to_string(feat));
		record = atol(recordTemplate->feature[feat].record);
		look = lookupTable[record];

		processed = get_flag(look);
		if (processed != processFlag) {
			//trace("processed != processFlag -> continue");
			continue;
		}

		subset = get_subset(look);

		if (isDoneOrNull(metadatas[subset*2]) || metadatas[subset*2]->bestSplit.decision.feature != feat)
			continue;

		//trace("Metadatas(" + to_string(i) + "): - check feature " + to_string(feat));

		category = get_category(look);
		value = atoi(recordTemplate->feature[feat].value);
		
		subset *= 2;

		//if (!isDoneOrNull(metadatas[subset])) {
			if (value <= metadatas[subset]->bestSplit.decision.refVal) {
				//trace(to_string(value) + " <= " + to_string(metadatas[subset]->bestSplit.decision.refVal) + " -> incr Subset " + to_string(subset));
				metadatas[subset]->increment(category);
			}
			else {
				subset += 1;
				//trace(to_string(value) + " !< " + to_string(metadatas[subset]->bestSplit.decision.refVal) + " -> incr Subset " + to_string(subset));
				metadatas[subset]->increment(category);
			}

			//trace("Lookup = " + to_string(look));
			//trace("Record = " + to_string(record));
			//trace("Subset = " + to_string(subset));
			//trace("Add Record " + to_string(record) + " to Subset " + to_string(subset) + "; " + to_string(value) + " < " +
			//	to_string(metadatas[subset]->bestSplit.decision.refVal));
		//}
			
		comb = combine(subset, !processed, category);
		//trace("Change Record " + to_string(record) + " from " + to_string(look) + " to " + to_string(comb));
		lookupTable[record] = comb;
	}
}

void printMetadata(std::vector<Metadata*>& metadatas) {
	trace("Metadatas: ");
	trace("size = " + to_string(metadatas.size()));

	for (int i = 0; i < metadatas.size(); i++) {
		if (metadatas[i] == NULL) {
			//trace("Metadatas(" + to_string(i) + ") == NULL");
		}
		else {
			trace("Metadatas(" + to_string(i) + ") == NOT NULL");
			trace(metadatas[i]->toString());
		}
	}
}

void printRecords(size_t* lookupTable, size_t pSubset) {
	size_t look, subset;
	
	trace("Subset " + to_string(pSubset) + ":");

	//for (size_t i = 0; i < tree::BPT_NUM_RECORDS; i++) {
	//	look = lookupTable[i];
	//	subset = get_subset(look);

	//	if (subset == pSubset) {
	//		std::stringstream ss;
	//		ss << "Record(" << std::setw(2) << std::setfill('0') << i << ") = 0x" << std::setw(4) << std::setfill('0') << std::hex << look;
	//		trace(ss.str());
	//	}
	//}
}

void printLookupTable(size_t* lookupTable) {
	size_t look, subset;

	trace("Lookup Table:");

	for (size_t i = 0; i < tree::BPT_NUM_RECORDS; i++) {
		look = lookupTable[i];
		subset = get_subset(look);
		std::stringstream ss;
		ss << "Record(" << std::setw(4) << std::setfill('0') << i << ") = 0x" << std::setw(4) << std::setfill('0') << std::hex << look;
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
		iterate(file, metadatas, buffer, buffer_size, line_buffer, line_size, recordTemplate, lookupTable, false, iteration, checkRecord);
		end = clock();
		elapsed = ((double)end - (double)start) / (double)CLOCKS_PER_SEC;
		printf("Iteration 1 took %05.2lf seconds.\n", elapsed);

		start = clock();
		processing = processMetadata(metadatas);
		end = clock();
		elapsed = ((double)end - (double)start) / (double)CLOCKS_PER_SEC;
		printf("processMetadata took %05.2lf seconds.\n", elapsed);

		if (!processing) {
			trace("processing = false -> break");
			break;
		}

		start = clock();
		iterate(file, metadatas, buffer, buffer_size, line_buffer, line_size, recordTemplate, lookupTable, true, iteration, assignRecord);
		end = clock();
		elapsed = ((double)end - (double)start) / (double)CLOCKS_PER_SEC;
		printf("Iteration 2 took %05.2lf seconds.\n", elapsed);

		start = clock();
		processing = postProcess(metadatas);
		end = clock();
		elapsed = ((double)end - (double)start) / (double)CLOCKS_PER_SEC;
		printf("postProcess took %05.2lf seconds.\n", elapsed);

		printMetadata(metadatas);
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

void iterate(FILE* file, std::vector<Metadata*>& metadata, char* buffer, int buffer_size, char* line_buffer, int line_size, 
	PlainRecord* recordTemplate, size_t* lookupTable, bool assign, int iteration,
	std::function<void(PlainRecord*,size_t*,std::vector<Metadata*>&, bool b)> func) {
	
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

			func(recordTemplate, lookupTable, metadata, flag);
		}
	}
}

bool processMetadata(std::vector<Metadata*>& metadatas) {

	trace("processMetadata()");

	bool done = true;

	int metad = -1;
	for (Metadata*& m : metadatas) {
		metad++;
		if (isDoneOrNull(m)) {
			//trace("Metadata(" + to_string(metad) + ") done -> continue");
			continue;
		}

		//trace("Metadata(" + to_string(metad) + ") not done -> process");

		bestSplit(m);

		done = false;

		if (m->nodeRef == NULL) {
			trace("Metadata " + to_string(m->count) + ") has no nodeRef!");
			printf("NodeRef is NULL! ERROR.\n");
			printf("Check the insertion mechanism for issues!\n");
		}

		*m->nodeRef = (tree::Node*) new tree::DecisionNode(m->bestSplit.decision);
		if (*m->nodeRef == NULL) {
			trace("Assigning DecisionNode failed!");
		}

		trace("DecisionNode: >" + (*m->nodeRef)->toString() + "<");
		m->clean();
	}

	if (done) {
		trace("Job DONE -> return");
		return false;
	}

	size_t size = metadatas.size();
	trace("Num Metadatas = " + to_string(size));

	for (int i = 1; i < size; i++) {
		if (metadatas[i * 2 - 2] == NULL) {
			metadatas.insert(metadatas.begin() + (i * 2 - 1), NULL);
		}
		else {
			metadatas.insert(metadatas.begin() + (i * 2 - 1), new Metadata());
			metadatas[i * 2 - 1]->bestSplit = metadatas[(i - 1) * 2]->bestSplit;
		}

		//if (metadatas[(i - 1) * 2]->done) {
		//	metadatas[i * 2 - 1]->done = true;
		//}
	}

	if (metadatas[metadatas.size() - 1] == NULL) {
		metadatas.insert(metadatas.end(), NULL);
	}
	else {
		metadatas.insert(metadatas.end(), new Metadata());
		size = metadatas.size();
		metadatas[size - 1]->bestSplit = metadatas[size - 2]->bestSplit;
	}

	trace("Num Metadatas after Insertion = " + to_string(size));

	if (size % 2 != 0) {
		printf("Error while processing metadata. size is odd!\n");
		exit(19);
	}

	//trace("All Metadatas:");
	//for (Metadata* m : metadatas) {
	//	trace(m->toString());
	//}

	//trace("Connect Nodes");
	size = metadatas.size();
	//trace("Metadatas.size() == " + to_string(size));
	for (int i = 0; i < size; i += 2) {
		if (isDoneOrNull(metadatas[i]))
			continue;

		//printf("true_branch pointer = %p\n", &(*metadatas[i]->nodeRef)->true_branch);
		//printf("false_branch pointer = %p\n", &(*metadatas[i]->nodeRef)->false_branch);

		//if (metadatas[i]->nodeRef == NULL) {
		//	trace("metadatas[" + to_string(i) + "]->nodeRef == NULL");
		//} 

		//if (metadatas[i + 1]->nodeRef == NULL) {
		//	trace("metadatas[" + to_string(i + 1) + "]->nodeRef == NULL");
		//}

		//trace("Connect Metadatas " + to_string(i) + " & " + to_string(i + 1) + " to false and true branch of Metadata " + to_string(i));
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

		//printf("metadatas[%d+1]->nodeRef = %p\n", i, metadatas[i + 1]->nodeRef);
		//printf("metadatas[%d]->nodeRef = %p\n", i, metadatas[i]->nodeRef);
	}

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
	trace("postProcess()");

	//int metad = 0;
	//trace("All Metadatas:");
	//for (Metadata* m : metadatas) {
	//	if (m->done)
	//		continue;

	//	trace("Metadata(" + to_string(metad++) + "):");
	//	trace(m->toString());
	//}

	//metad = 0;
	bool done = true;
	for (Metadata*& m : metadatas) {
		if (isDoneOrNull(m))
			continue;

		//trace("handle Metadata(" + to_string(metad++) + ")");
		done = false;
		impurity(m);

		//for test dataset
		if(m->uncertainty == 0) {
		//if (m->uncertainty <= tree::BPT_STOP_EVALUATION_IMPURITY || m->totalNumRecords <= tree::BPT_STOP_EVALUATION_LIMIT) {
			//trace("numRecords = " + to_string(m->totalNumRecords));
			*m->nodeRef = (tree::Node*) new tree::ResultNode(as_result(m));
			trace("ResultNode: " + (*m->nodeRef)->toString());
			m->done = true;
			delete m;
			m = NULL;
		}
	}

	return !done;
}