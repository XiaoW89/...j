#pragma once
#include "utils.h"

class JDA
{
private:
	_Parameters _pm;
	cv::Mat_<float> _weights_neg;
	cv::Mat_<float> _weights_pos;

	std::list<float> _score_neg;
	std::list<float> _score_pos;

	Dt JDA::GeNegSamp(const _MyData* const md);
	

public:
	JDA();
	JDA(const _Parameters&);
	~JDA();

	void train(const _MyData* md);

};

