#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H
#include "utils.h"
#include <set>

enum CorR
{
	CLASSIFICATION = 0, REGRESSION
};

class Node {
public:
	int _leaf_identity; // used only when it is leaf node, and is unique among the tree
	Node* _left_child;
	Node* _right_child;
	int _samples;
	bool _is_leaf;
	int _depth; // recording current depth
	double _threshold;
	bool _thre_changed;
	CorR _cor;
	double _cscore;
	FeatureLocations _feature_locations;
	Node(Node* left, Node* right, double thres, bool leaf);
	Node(Node* left, Node* right, double thres);
	Node();

	float _Precision;
	float _Recall;
	float _Fscore;
	float _theta; //used for removing samples whos _cscore<_theta, and this value is acquired according to orecision-recall rate;
};

class RandomForest {
public:
	int _stage;
	int _local_features_num;
	int _landmark_index;
	int _tree_depth;
	int _trees_num_per_forest;
	double _local_radius;
	int _all_leaf_nodes;
	//cv::Mat_<double> mean_shape_;
	std::vector<Node*> _trees;
	std::vector<FeatureLocations> _local_position; // size = param_.local_features_num
	std::vector<cv::Mat_<double> >* _regression_targets;

	bool TrainForest(MYDATA* const md, std::deque<DT>& p_dt, std::deque<DT>& n_dt,
		const std::vector<cv::Mat_<double> >& rotations_negsample, const std::vector<cv::Mat_<double> >& rotations_possample,
		const std::vector<double>& scales_negsample, const std::vector<double>& scales_possample);

	Node* BuildCRTree(std::set<int>& selected_ft_indexes,int current_depth, std::deque<DT>& p_dt, 
		std::deque<DT>& n_dt);

	int FindSplitFeature(Node* node, std::set<int>& selected_ft_indexes, std::deque<DT>& left_pos,
		std::deque<DT>& left_neg, std::deque<DT>& right_pos, std::deque<DT>& right_neg,
		CorR corr, const std::deque<DT>& p_dt, const std::deque<DT>& n_dt);

	cv::Mat_<double> GetBinaryFeatures(const cv::Mat_<double>& image,
		const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale);

	int MarkLeafIdentity(Node* node, int count);

	int GetNodeOutput(Node* node, const cv::Mat_<double>& image,
		const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale);

	//predict()
	int GetBinaryFeatureIndex(int tree_index, const cv::Mat_<double>& image,
	const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale);

	RandomForest();

	RandomForest(PARAMETERS& param, int landmark_index, int stage, std::vector<cv::Mat_<double> >& regression_targets);

	void WriteTree(Node* p, std::ofstream& fout);

	Node* ReadTree(std::ifstream& fin);

	void SaveRandomForest(std::ofstream& fout);

	void LoadRandomForest(std::ifstream& fin);

	void GeneratePixelDiff(MYDATA* const md, std::deque<DT>& dt);

private:
	CorR Split_Type(const int stage);

	void AssignCScore_Node(Node* nd, std::deque<DT>& p_dt, std::deque<DT>& n_dt);

	void getCscore_singleTress(Node* nd, DT& dt);
};

#endif
