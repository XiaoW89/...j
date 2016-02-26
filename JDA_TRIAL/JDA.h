#pragma once
#include "utils.h"
#include "randomforest.h"

class JDA
{
private:
	PARAMETERS _pm;
	cv::Mat_<float> _weights_neg;
	cv::Mat_<float> _weights_pos;

	std::deque<float> _score_neg;
	std::deque<float> _score_pos;

	DT GeNegSamp(MYDATA* const md);
	void AsignWeight(DT& dt){ dt._weight = exp(-1 * dt._lable* dt._score); };
	void LearnCRTrees(const std::deque<DT>& p_dt, const std::deque<DT>& n_dt, const PARAMETERS& pm);
	

public:
	JDA();
	JDA(const PARAMETERS&);
	~JDA();

	void train(MYDATA* const md);

};

