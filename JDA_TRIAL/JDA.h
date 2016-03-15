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

	std::vector<cv::Mat_<float>>_shape_param_set;



public:

	JDA();
	JDA(const PARAMETERS&);
	~JDA();

	void trainJDA(MYDATA* const md);

};

