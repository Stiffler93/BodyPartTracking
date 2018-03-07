#include "BoundlessTrainingUtils.h"
#include <string>

using std::string;
using std::to_string;

FeatureStats::FeatureStats()
{
}

FeatureStats::FeatureStats(int feature, Metadata* metadata) {
	for (int i = 0; i < NUM_CATEGORIES; i++)
		recordsPassed[i] = 0;

	this->metadata = metadata;
	bestSplit.decision.feature = feature;
}

void FeatureStats::nextValue(short value, char category) {
	bool replace = false;
	if (value == valueOfBestSplit)
		replace = true;

	totalNumRecordsPassed++;
	recordsPassed[category]++;

	if (value != bestSplit.decision.refVal
		&& bestSplit.gain > overallBestSplit.gain) {
		overallBestSplit = bestSplit;
		//if (bestSplit.decision.feature == 4 || bestSplit.decision.feature == 5) {
		//	trace("Feature(" + to_string(bestSplit.decision.feature) + "): overallBestSplit.gain(" + to_string(overallBestSplit.gain) + ") > bestSplit.gain(" + to_string(bestSplit.gain) + ")");
		//}
	}

	float gain = infoGain(this, metadata);
	if (gain > bestSplit.gain || replace) {
		valueOfBestSplit = value;
		//if (bestSplit.decision.feature == 4 || bestSplit.decision.feature == 5)
		//	trace("Feature(" + to_string(bestSplit.decision.feature) + "): gain(" + to_string(gain) + ") > bestSplit.gain(" + to_string(bestSplit.gain) + ")");
		bestSplit.gain = gain;
		bestSplit.decision.refVal = value;
	}
}

tree::BestSplit FeatureStats::getBestSplit()
{
	if (overallBestSplit.gain > bestSplit.gain) {
		//trace("overallBestSplit.gain(" + to_string(overallBestSplit.gain) + " > bestSplit.gain(" + to_string(bestSplit.gain) + ")");
		return overallBestSplit;
	}

	//trace("return bestSplit.gain(" + to_string(bestSplit.gain) + ")");
	return bestSplit;
}

Metadata::Metadata() {
	count = ++counter;
	for (int i = 0; i < tree::BPT_NUM_FEATURES; i++) {
		stats[i] = FeatureStats(i, this);
	}
}

void Metadata::increment(char category) {
	//trace("Increment category " + to_string(category));
	totalNumRecords++;
	recordsPerCategory[category]++;
	//trace("New totalNumRecords = " + to_string(totalNumRecords));
	//trace("New records counted for category = " + to_string(recordsPerCategory[category]));
}

string Metadata::toString() {
	std::stringstream ss;
	ss << "Metadata: " << count << std::endl;
	ss << "done? " << done << std::endl;
	ss << "totalNumRecords = " << to_string(totalNumRecords) << std::endl;
	for (int c = 0; c < NUM_CATEGORIES; c++) {
		ss << "Category(" << to_string(c) << ") = " << to_string(recordsPerCategory[c]) << std::endl;
	}
	ss << "Uncertainty = " << uncertainty << std::endl;
	ss << "BestSplit = >" << to_string(bestSplit.decision.feature) << "," << to_string(bestSplit.decision.refVal) << "<" << std::endl;
	if (nodeRef != NULL && *nodeRef != NULL) {
		ss << "node: " << (*nodeRef)->toString() << std::endl;
	}
	else if (nodeRef != NULL && *nodeRef == NULL) {
		ss << "nodeRef points to node, but node not initialized yet!" << std::endl;
	}
	else {
		ss << "No Node" << std::endl;
	}
	ss << std::endl;

	return ss.str();
}

void Metadata::clean() {
	totalNumRecords = 0;
	for (int i = 0; i < tree::BPT_NUM_FEATURES; i++) {
		stats[i] = FeatureStats(i, this);
	}
	for (int c = 0; c < NUM_CATEGORIES; c++) {
		recordsPerCategory[c] = 0;
	}
	bestSplit.gain = 0;
	uncertainty = 0;
}

int Metadata::counter = 0;

size_t get_subset(size_t val) {
	return (val & MASK_SUBSET) >> BITS_VALUE;
}

char get_category(size_t val) {
	return (char)(val & MASK_VALUE);
}

bool get_flag(size_t val) {
	return (bool)((val & FLAG_PROCESSED) >> MOVE_FLAG);
}

size_t combine(size_t subset, bool flag, char value) {
	return ((subset << BITS_VALUE) & MASK_SUBSET) | (value & MASK_VALUE) | ((char)flag << MOVE_FLAG);
}

void impurity(Metadata *metadata) {
	//trace("impurity(Metadata)");
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
	//trace("uncertainty = " + to_string(metadata->uncertainty));
}

float impurity(FeatureStats* stats) {
	//trace("impurity(Featurestats)");
	double uncertainty = 1;
	size_t totalNumber = stats->totalNumRecordsPassed;
	//trace("TotalNumber = " + to_string(totalNumber));
	if (totalNumber == 0)
		return 0;

	for (short i = 0; i < NUM_CATEGORIES; i++) {
		size_t number = stats->recordsPassed[i];
		double reduce = pow(((double)number / (double)totalNumber), 2);
		//trace("Reduce = " + to_string(reduce));
		uncertainty -= reduce;
		//trace("Decreased uncertainty = " + to_string(uncertainty));
	}

	//trace("uncertainty = " + to_string(uncertainty));

	return (float)uncertainty;
}

float impurity(FeatureStats* stats, Metadata* metadata) {
	//trace("impurity(FeatureStats, Metadata)");
	double uncertainty = 1;
	size_t totalNumber = metadata->totalNumRecords - stats->totalNumRecordsPassed;
	//trace("TotalNumber = " + to_string(totalNumber));

	for (short i = 0; i < NUM_CATEGORIES; i++) {
		size_t number = metadata->recordsPerCategory[i] - stats->recordsPassed[i];
		double reduce = pow(((double)number / (double)totalNumber), 2);
		//trace("Reduce = " + to_string(reduce));
		uncertainty -= reduce;

		//trace("Decreased uncertainty = " + to_string(uncertainty));
	}

	//trace("uncertainty = " + to_string(uncertainty));

	return (float)uncertainty;
}

float infoGain(FeatureStats* stats, Metadata* metadata) {
	//trace("infoGain()");
	if (stats->totalNumRecordsPassed == 0 || stats->totalNumRecordsPassed == metadata->totalNumRecords)
		return 0;
	float p = (float)stats->totalNumRecordsPassed / (float)metadata->totalNumRecords;
	float imp1 = impurity(stats);
	float imp2 = impurity(stats, metadata);
	float infoGain = metadata->uncertainty - p * imp1 - (1 - p) * imp2;
	//trace("totalNumRecordsPassed = " + to_string(stats->totalNumRecordsPassed) + ", totalNumRecords = " + to_string(metadata->totalNumRecords));
	//trace("infoGain = " + to_string(metadata->uncertainty) + " - " + to_string(p) + " * " + to_string(imp1) + " - " + to_string(1 - p) + " * " +
	//to_string(imp2) + " = " + to_string(infoGain));
	return infoGain;
}

void bestSplit(Metadata *metadata) {
	if (isDoneOrNull(metadata))
		return;

	//trace("Best Split:");

	//int i = 0;
	//for (FeatureStats s : metadata->stats) {
	//	trace("FeatureStats(" + to_string(i++) + "): gain = " + to_string(s.getBestSplit().gain) + ", dec = " + to_string(s.getBestSplit().decision.feature)
	//		+ "|" + to_string(s.getBestSplit().decision.refVal));
	//}

	const int FACTOR = 1000000;
	tree::BestSplit bestSplit;
	int newGain, oldGain;

	for (int f = 0; f < tree::BPT_NUM_FEATURES; f++) {
		newGain = (int)(metadata->stats[f].getBestSplit().gain * FACTOR);
		oldGain = (int)(bestSplit.gain * FACTOR);
		//if (metadata->stats[f].getBestSplit().gain >= bestSplit.gain) {
		// for comparison with parallel!
		if(newGain >= oldGain) {
			bestSplit = metadata->stats[f].getBestSplit();
		}
	}

	metadata->bestSplit = bestSplit;
	//trace("Metadatas BestSplit = <" + to_string(bestSplit.gain) + "|" + to_string(bestSplit.decision.feature) + ":" + to_string(bestSplit.decision.refVal)
	//	+ ">");
}

bool isDoneOrNull(Metadata*& m) {
	if (m == NULL)
		return true;

	if (m->done) {
		delete m;
		m = NULL;
		return true;
	}

	return false;
}