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
	FeatureLocations _feature_locations;
	Node(Node* left, Node* right, double thres, bool leaf);
	Node(Node* left, Node* right, double thres);
	Node();
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

	Node* BuildTree(std::set<int>& selected_indexes_pos, std::set<int>& selected_indexes_neg,
		std::vector<int>& images_indexes_pos, std::vector<int>& images_indexes_neg, int current_depth,
		std::deque<DT>& p_dt, std::deque<DT>& n_dt);

	int FindSplitFeature(Node* node, std::set<int>& selected_indexes,
		cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int>& left_indexes, std::vector<int>& right_indexes);

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

	CorR Split_Type(const int stage);
};

#endif
