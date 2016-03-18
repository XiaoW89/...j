#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H
#include "utils.h"
#include <set>



class RandomForest {
public:
	
	int _stage;
	int _local_features_num;
	int _landmark_index;
	int _tree_depth;
	int _trees_num_per_forest;
	double _local_radius;
	int _all_leaf_nodes; // used for labeling the index of leaf node in current stage
	std::vector<std::vector<Node*>> trees_;
	//cv::Mat_<double> mean_shape_;
	std::vector<std::vector<FeatureLocations>> _local_position; // size = param_.local_features_num

	bool TrainForest(MYDATA* const md, const PARAMETERS& pm,
		const std::vector<cv::Mat_<float>>&shape_param_set, std::deque<DT*>& p_dt,
		std::deque<DT*>& n_dt, std::vector<RandomForest>&cascade);

	Node* BuildCRTree(std::set<int>& selected_ft_indexes,int current_depth, std::deque<DT*>& p_dt, 
		std::deque<DT*>& n_dt);

	int FindSplitFeature(Node* node, std::set<int>& selected_ft_indexes, std::deque<DT*>& left_pos,
		std::deque<DT*>& left_neg, std::deque<DT*>& right_pos, std::deque<DT*>& right_neg,
		CorR corr, const std::deque<DT*>& p_dt, const std::deque<DT*>& n_dt);

	void LearnShapeIncrement(const std::vector<std::vector<Node*>>&cascade, std::deque<DT*>& p_dt);
	RandomForest();
	RandomForest(PARAMETERS& param,  int stage);
	void GeneratePixelDiff(MYDATA* const md, std::deque<DT*>& dt, const std::vector<FeatureLocations>& fl);

private:

	CorR Split_Type(const int stage);

	void AssignCScore_Node(Node* nd, std::deque<DT*>& p_dt, std::deque<DT*>& n_dt);

	void getCscore_singleTress(const Node* nd, DT* dt);

	bool getCscore_wholeTress(std::vector<RandomForest>&cascade,
		const std::vector<cv::Mat_<float>>&_shape_param_set, MYDATA* md, DT* dt);

	//int GetLeafIndex_singleTress(Node* nd, DT* dt);


};

void getlocallbf(const Node* nd, DT* dt);
void GetGlobalLBF(MYDATA* md, RandomForest& rf, DT* dt);

#endif
