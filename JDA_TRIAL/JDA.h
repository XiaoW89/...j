#pragma once
#include "utils.h"
#include "randomforest.h"
#include <linear.h>

class JDA
{
private:
	PARAMETERS _pm;
	cv::Mat_<float> _weights_neg;
	cv::Mat_<float> _weights_pos;

	std::deque<float> _score_neg;
	std::deque<float> _score_pos;

	void getlocallbf(const Node* nd, DT* dt);
	void GetGlobalLBF(MYDATA* md, RandomForest& rf, DT* dt);
	void UpdateShape(const cv::Mat_<float>& weights, DT* dt);


public:
	JDA();
	JDA(const PARAMETERS&);
	~JDA();

	void trainJDA(MYDATA* const md);

};

