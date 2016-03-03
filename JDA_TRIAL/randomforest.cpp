#include "randomforest.h"
#include <time.h>
#include <algorithm>
#include <stack>
Node::Node()
{
	_left_child = NULL;
	_right_child = NULL;
	_is_leaf = false;
	_threshold = 0.0;
	_leaf_identity = -1;
	_samples = -1;
	_thre_changed = false;
	_cor = CLASSIFICATION;  
	
}

Node::Node(Node* left, Node* right, double thres)
{
	Node(left, right, thres, false);
}

Node::Node(Node* left, Node* right, double thres, bool leaf)
{
	_left_child = left;
	_right_child = right;
	_is_leaf = leaf;
	_threshold = thres;
	//offset_ = cv::Point2f(0, 0);
}

bool RandomForest::TrainForest(MYDATA* const md,  std::deque<DT>& p_dt,  std::deque<DT>& n_dt,
	const std::vector<cv::Mat_<double> >& rotations_negsample, const std::vector<cv::Mat_<double> >& rotations_possample, 
	const std::vector<double>& scales_negsample, const std::vector<double>& scales_possample)
{
	int n_sample_neg = n_dt.size();
	int n_sample_pos = p_dt.size();

	_local_position.clear();
	_local_position.resize(_local_features_num);
	for (int i = 0; i < _local_features_num; i++)
	{
		double x, y;
		do{
			x = RandNumberUniform<double>(-_local_radius, _local_radius);
			y = RandNumberUniform<double>(-_local_radius, _local_radius);
		} while (x*x + y*y > _local_radius*_local_radius);
		cv::Point2f a(x, y);

		do{
			x = RandNumberUniform<double>(-_local_radius, _local_radius);
			y = RandNumberUniform<double>(-_local_radius, _local_radius);
		} while (x*x + y*y > _local_radius*_local_radius);
		cv::Point2f b(x, y);

		_local_position[i] = FeatureLocations(a, b);
	}
	
	//Extract pixel difference feature
	std::cout << "get pixel differences" << std::endl;
	GeneratePixelDiff(md, p_dt);
	GeneratePixelDiff(md, n_dt);

	// train Random Forest
	// construct each tree in the forest
	_trees.clear();
	_all_leaf_nodes = 0;
	for (int i = 0; i < _trees_num_per_forest; i++)
	{
		std::set<int> selected_indexes; //used for storing features' index when perform node-split 
		Node* root = BuildCRTree(selected_indexes,  0, p_dt, n_dt);
		
		//Update classification score for each sample
		std::set<float>score_set;
		for (int j = 0; j < p_dt.size(); j++)
		{
			getCscore_singleTress(root, p_dt[j]);
			score_set.insert(p_dt[j]._score);
		}
		for (int j = 0; j < n_dt.size(); j++)
		{
			getCscore_singleTress(root, n_dt[j]);
			score_set.insert(n_dt[j]._score);
		}


		//Determing the bias theta according to a preset precision-recall condition
		root->_Precision = root->_Recall = 0.0;
		root->_Fscore = -DBL_MAX;
		root->_theta = 0;

		for (std::set<float>::const_iterator it; it != score_set.cend(); it++)
		{
			float threshold = *it;
			float fp = 0.0; int tp = 0.0;
			for (int t = 0; t < n_sample_pos; t++)
			{
				if (p_dt[t]._score >= threshold)
					tp += 1;
			}
			for (int t = 0; t < n_sample_neg; t++)
			{
				if (n_dt[t]._score >= threshold)
					fp += 1;
			}
			float Precision = (tp) / (float)(tp + fp);
			float Recall = (tp ) / (float)(n_sample_pos);
			float fscore = 2 * Precision*Recall / (Precision + Recall);
			if (fscore > root->_Fscore)
			{
				root->_Precision = Precision;
				root->_Recall = Recall;
				root->_Fscore = fscore;
				root->_theta = threshold;
			}
		}

		//Removing samples whos classification score less than theta



		_trees.push_back(root);
	}

	/*int count = 0;
	for (int i = 0; i < trees_num_per_forest_; i++){
		Node* root = trees_[i];
		count = MarkLeafIdentity(root, count);
	}
	all_leaf_nodes_ = count;*/
	return true;
}


Node* RandomForest::BuildCRTree(std::set<int>& selected_ft_indexes, int current_depth, std::deque<DT>& p_dt,
	std::deque<DT>& n_dt)
{
	int n_sample_neg = n_dt.size();
	int n_sample_pos = p_dt.size();

	if ((n_sample_pos > 0) & (n_sample_neg > 0))// the node may not split under some cases
	{ 
		//Decide either CLASSIFICATION or REGRESSION node
		CorR type_split = Split_Type(_stage);

		//Construct tree iteratively
		Node* node = new Node();
		node->_depth = current_depth;
		node->_samples = n_sample_pos+n_sample_neg;
	
		std::deque<DT> left_image_pos, right_image_pos,
			left_image_neg, right_image_neg;
		if (current_depth == _tree_depth){ // the node reaches max depth
			node->_is_leaf = true;
			node->_leaf_identity = _all_leaf_nodes;
			_all_leaf_nodes++;
			AssignCScore_Node(node, p_dt, n_dt); //Assign classification score for leaf node
			return node;
		}

		int ret = FindSplitFeature(node, selected_ft_indexes, left_image_pos, left_image_neg,
			right_image_pos,right_image_neg, type_split, p_dt, n_dt);

	
		// actually it won't enter the if block, when the random function is good enough
		if (ret == 1){ // the current node contain all sample when reaches max variance reduction, it is leaf node
			node->_is_leaf = true;
			node->_leaf_identity = _all_leaf_nodes;
			_all_leaf_nodes++;
			AssignCScore_Node(node, p_dt, n_dt);
			return node;
		}

		node->_left_child = BuildCRTree(selected_ft_indexes, current_depth + 1, left_image_pos, left_image_neg);
		node->_right_child = BuildCRTree(selected_ft_indexes, current_depth + 1, right_image_pos, right_image_neg);

		return node;
	}
	else{ // this case is not possible in this data structure
		return NULL;
	}
}


int RandomForest::FindSplitFeature(Node* node, std::set<int>& selected_ft_indexes, std::deque<DT>& left_pos,
	std::deque<DT>& left_neg, std::deque<DT>& right_pos, std::deque<DT>& right_neg,
	CorR corr, const std::deque<DT>& p_dt, const std::deque<DT>& n_dt)
{
	std::vector<int> val;
	//cv::Mat_<int> sorted_fea;
	int threshold; 
	double var = -DBL_MAX;
	double entrp = -DBL_MAX;
	int feature_index = -1;

	//int j = 0, tmp_index;

	int n_p = p_dt.size();
	int n_n = n_dt.size();

	std::deque<DT> tp_left_pos, tp_left_neg, tp_right_pos, tp_right_neg;
	for (int j = 0; j < _local_features_num; j++)
	{
		if (selected_ft_indexes.find(j) == selected_ft_indexes.end()) //如果j不在容器里面就执行操作
		{
		
			tp_left_pos.clear(); tp_left_neg.clear(); tp_right_neg.clear(); tp_right_pos.clear();
			//I am training regression nodes with positive samples only
			if (REGRESSION == corr) //For Regression Node
			{
				double var_lc = 0.0, var_rc = 0.0, var_red = 0.0;
				double Ex_2_lc = 0.0, Ex_lc = 0.0, Ey_2_lc = 0.0, Ey_lc = 0.0;
				double Ex_2_rc = 0.0, Ex_rc = 0.0, Ey_2_rc = 0.0, Ey_rc = 0.0;

				// random generate threshold (****Need be modifed)
				std::vector<int> data; 
				data.reserve(n_p);
				for (int i = 0; i < n_p; i++){
					data.push_back(p_dt[i]._pixDiffFeat(j));
				}
				std::sort(data.begin(), data.end());
				int tmp_index = floor((int)(n_p*(0.5 + 0.9*(RandNumberUniform<float>(0.0, 1.0) - 0.5))));
				int tmp_threshold = data[tmp_index];
				for (int i = 0; i < n_p; i++)
				{
					if ((p_dt[i]._pixDiffFeat(j) < tmp_threshold))
					{
						tp_left_pos.push_back(p_dt[i]);
						// do with regression target
						double value = _regression_targets->at(i)(_landmark_index, 0);
						Ex_2_lc += pow(value, 2);//左集合中x^2的期望
						Ex_lc += value; //左集合中x的期望
						value = _regression_targets->at(i)(_landmark_index, 1);
						Ey_2_lc += pow(value, 2);
						Ey_lc += value;
					}
					else
					{
						tp_right_pos.push_back(p_dt[i]);
						double value = _regression_targets->at(i)(_landmark_index, 0);
						Ex_2_rc += pow(value, 2);
						Ex_rc += value;
						value = _regression_targets->at(i)(_landmark_index, 1);
						Ey_2_rc += pow(value, 2);
						Ey_rc += value;
					}
				}

				if (tp_left_pos.size() == 0){ var_lc = 0.0; }
				else
				{
					var_lc = Ex_2_lc / tp_left_pos.size() - pow(Ex_lc / tp_left_pos.size(), 2)
						+ Ey_2_lc / tp_left_pos.size() - pow(Ey_lc / tp_left_pos.size(), 2);  //x坐标的方差加上y坐标的方差
				}

				if (tp_right_pos.size() == 0){ var_rc = 0.0; }
				else
				{
					var_rc = Ex_2_rc / tp_right_pos.size() - pow(Ex_rc / tp_right_pos.size(), 2)
						+ Ey_2_rc / tp_right_pos.size() - pow(Ey_rc / tp_right_pos.size(), 2);
				}

				var_red = -var_lc*tp_left_pos.size() - var_rc*tp_right_pos.size();//这四个量均是非负，因此var_red非正
				if (var_red > var) //这里用的是最小均方差LSD，主要是希望类内方差小
				{
					var = var_red;
					threshold = tmp_threshold;
					feature_index = j;
					tp_left_neg = tp_right_neg =  n_dt; //原封不动的送入下一层
				}
			}
			else //For classification node
			{
				// random generate threshold (****Need be modifed)
				std::vector<int> data;
				data.reserve(p_dt.size() + n_dt.size());
				for (int i = 0; i < p_dt.size(); i++){
					data.push_back(p_dt[i]._pixDiffFeat(j));
				}
				for (int i = 0; i < n_dt.size(); i++){
					data.push_back(n_dt[i]._pixDiffFeat(j));
				}

				std::sort(data.begin(), data.end());
				int tmp_index = floor((int)(data.size()*(0.5 + 0.9*(RandNumberUniform<float>(0.0, 1.0) - 0.5))));
				int tmp_threshold = data[tmp_index];

				for (int i = 0; i < n_p; i++)
				{
					if ((p_dt[i]._pixDiffFeat(j) < tmp_threshold)){ tp_left_pos.push_back(p_dt[i]); }
					else{ tp_right_pos.push_back(p_dt[i]); }
				}
				for (int i = 0; i < n_n; i++)
				{
					if ((n_dt[i]._pixDiffFeat(j) < tmp_threshold)){ tp_left_neg.push_back(n_dt[i]); }
					else{ tp_right_neg.push_back(n_dt[i]); }
				}

				int n_n_l = tp_left_neg.size();
				int n_n_r = tp_right_neg.size();
				int n_p_l = tp_left_pos.size();
				int n_p_r = tp_right_pos.size();

				double entrp_root = -(n_n / (n_n + n_p))*log((n_n / (n_n + n_p))) - (n_p / (n_n + n_p))*log((n_p / (n_n + n_p)));
				double entrp_l = -(n_n_l / (n_n_l + n_p_l))*log((n_n_l / (n_n_l + n_p_l))) - (n_p_l / (n_n_l + n_p_l))*log((n_p_l / (n_n_l + n_p_l)));
				double entrp_r = -(n_n_r / (n_n_r + n_p_r))*log((n_n_r / (n_n_r + n_p_r))) - (n_p_r / (n_n_r + n_p_r))*log((n_p_r / (n_n_r + n_p_r)));

				double inform_gain = entrp_root - (n_n_l + n_p_l)*entrp_l / (n_n + n_p) - (n_n_r + n_p_r)*entrp_r / (n_n + n_p);

				if (inform_gain > entrp) //这里用的是最小均方差LSD，主要是希望类内方差小
				{
					entrp = inform_gain;
					threshold = tmp_threshold;
					feature_index = j;
					left_neg = tp_left_neg;
					left_pos = tp_left_pos;
					right_neg = tp_right_neg;
					right_pos = tp_right_pos;
				}
			}
		}
	}

	if (feature_index != -1) // actually feature_index will never be -1 
	{
		if (0 == (left_neg.size() + left_pos.size()) ||
			0 == (right_neg.size() + right_pos.size()))
		{
			std::cout << "Waring: this node contain all the samples" << std::endl;
			node->_is_leaf = true; // the node can contain all the samples
			return 1;
		}

		node->_threshold = threshold;
		node->_thre_changed = true;
		node->_feature_locations = _local_position[feature_index];
		selected_ft_indexes.insert(feature_index);
		if (REGRESSION == corr) 
			node->_cor = REGRESSION;
		else 
			node->_cor = CLASSIFICATION;

		return 0;
	}
	return -1;
}


int RandomForest::MarkLeafIdentity(Node* node, int count){
	std::stack<Node*> s;
	Node* p_current = node; 
	
	if (node == NULL){
		return count;
	}
	// the node in the tree is either leaf node or internal node that has both left and right children
	while (1)//p_current || !s.empty())
	{
		
		if (p_current->_is_leaf)
		{
			p_current->_leaf_identity = count;
			count++;
			if (s.empty())
			{
				return count;
			}
			p_current = s.top()->_right_child;
			s.pop();
		}
		else
		{
			s.push(p_current);
			p_current = p_current->_left_child;
		}
		
		/*while (!p_current && !s.empty()){
			p_current = s.top();
			s.pop();
			p_current = p_current->right_child_; 
		}*/
	}
	
}

cv::Mat_<double> RandomForest::GetBinaryFeatures(const cv::Mat_<double>& image,
	const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale)
{
	cv::Mat_<double> res(1, _all_leaf_nodes, 0.0);
	for (int i = 0; i < _trees_num_per_forest; i++){
		Node* node = _trees[i];
		while (!node->_is_leaf){
			int direction = GetNodeOutput(node, image, bbox, current_shape, rotation, scale);
			if (direction == -1){
				node = node->_left_child;
			}
			else{
				node = node->_right_child;
			}
		}
		res(0, node->_leaf_identity) = 1.0;
	}
	return res;
}

int RandomForest::GetBinaryFeatureIndex(int tree_index, const cv::Mat_<double>& image,
	const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale){
	Node* node = _trees[tree_index];
	while (!node->_is_leaf){
		FeatureLocations& pos = node->_feature_locations;
		double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
		double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
		delta_x = scale*delta_x*bbox.width / 2.0;
		delta_y = scale*delta_y*bbox.height / 2.0;
		int real_x = delta_x + current_shape(_landmark_index, 0);
		int real_y = delta_y + current_shape(_landmark_index, 1);
		real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
		real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
		int tmp = (int)image(real_y, real_x); //real_y at first

		delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
		delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
		delta_x = scale*delta_x*bbox.width / 2.0;
		delta_y = scale*delta_y*bbox.height / 2.0;
		real_x = delta_x + current_shape(_landmark_index, 0);
		real_y = delta_y + current_shape(_landmark_index, 1);
		real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
		real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
		if ((tmp - (int)image(real_y, real_x)) < node->_threshold){
			node = node->_left_child;// go left
		}
		else{
			node = node->_right_child;// go right
		}
	}
	return node->_leaf_identity;
}


int RandomForest::GetNodeOutput(Node* node, const cv::Mat_<double>& image,
	const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale){
	
	FeatureLocations& pos = node->_feature_locations;
	double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
	double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
	delta_x = scale*delta_x*bbox.width / 2.0;
	delta_y = scale*delta_y*bbox.height / 2.0;
	int real_x = delta_x + current_shape(_landmark_index, 0);
	int real_y = delta_y + current_shape(_landmark_index, 1);
	real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
	real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
	int tmp = (int)image(real_y, real_x); //real_y at first

	delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
	delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
	delta_x = scale*delta_x*bbox.width / 2.0;
	delta_y = scale*delta_y*bbox.height / 2.0;
	real_x = delta_x + current_shape(_landmark_index, 0);
	real_y = delta_y + current_shape(_landmark_index, 1);
	real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
	real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
	if ((tmp - (int)image(real_y, real_x)) < node->_threshold){
		return -1; // go left
	}
	else{
		return 1; // go right
	}

}

RandomForest::RandomForest(PARAMETERS& param, int landmark_index, int stage, std::vector<cv::Mat_<double> >& regression_targets){
	_stage = stage;
	_local_features_num = param._n_splitFeatures;
	_landmark_index = landmark_index;
	_tree_depth = param._n_deepth;
	_trees_num_per_forest = param._n_childTress;
	_local_radius = param._radius[_stage];
	//mean_shape_ = param.mean_shape_;
	_regression_targets = &regression_targets; // get the address pointer, not reference
}

RandomForest::RandomForest(){
	
}

void RandomForest::SaveRandomForest(std::ofstream& fout){
	fout << _stage << " "
		<< _local_features_num << " "
		<< _landmark_index << " "
		<< _tree_depth << " "
		<< _trees_num_per_forest << " "
		<< _local_radius << " "
		<< _all_leaf_nodes << " "
		<< _trees.size() << std::endl;
	for (int i = 0; i < _trees.size(); i++){
		Node* root = _trees[i];
		WriteTree(root, fout);
	}
}

void RandomForest::WriteTree(Node* p, std::ofstream& fout){
	if (!p){
		fout << "#" << std::endl;
	}
	else{
		fout <<"Y" << " "
			<< p->_threshold << " " 
			<< p->_is_leaf << " "
			<< p->_leaf_identity << " "
			<< p->_depth << " "
			<< p->_feature_locations.start.x << " "
			<< p->_feature_locations.start.y << " "
			<< p->_feature_locations.end.x << " "
			<< p->_feature_locations.end.y << std::endl;
		WriteTree(p->_left_child, fout);
		WriteTree(p->_right_child, fout);
	}
}

Node* RandomForest::ReadTree(std::ifstream& fin){
	std::string flag;
	fin >> flag;
	if (flag == "Y"){
		Node* p = new Node();
		fin >> p->_threshold
			>> p->_is_leaf
			>> p->_leaf_identity
			>> p->_depth
			>> p->_feature_locations.start.x
			>> p->_feature_locations.start.y
			>> p->_feature_locations.end.x
			>> p->_feature_locations.end.y;
		p->_left_child = ReadTree(fin);
		p->_right_child = ReadTree(fin);
		return p;
	}
	else{
		return NULL;
	}
}

void RandomForest::LoadRandomForest(std::ifstream& fin){
	
	int tree_size;
	fin >> _stage
		>> _local_features_num
		>> _landmark_index
		>> _tree_depth
		>> _trees_num_per_forest
		>> _local_radius
		>> _all_leaf_nodes
		>> tree_size;
	std::string start_flag;
	_trees.clear();
	for (int i = 0; i < tree_size; i++){
		Node* root = ReadTree(fin);
		_trees.push_back(root);
	}
}

void RandomForest::GeneratePixelDiff(MYDATA* const md, std::deque<DT>& dt)
{
	for (int i = 0; i < dt.size(); i++)
	{
		//getSimilarityTransform(ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]),mean_shape_, rotation, scale);

		for (int j = 0; j < _local_features_num; j++){
			FeatureLocations pos = _local_position[j];
			double delta_x = dt[i]._rotation(0, 0)*pos.start.x + dt[i]._rotation(0, 1)*pos.start.y;//旋转后的x
			double delta_y = dt[i]._rotation(1, 0)*pos.start.x + dt[i]._rotation(1, 1)*pos.start.y;//旋转后的y
			delta_x = dt[i]._scale*delta_x*dt[i]._bbox.width / 2.0;//在框内的坐标
			delta_y = dt[i]._scale*delta_y*dt[i]._bbox.height / 2.0;
			int real_x = delta_x + dt[i]._prdshape(_landmark_index, 0);//在整个图片内的坐标
			int real_y = delta_y + dt[i]._prdshape(_landmark_index, 1);
			real_x = std::max(0, std::min(real_x, static_cast<int>(md->_imagsProperty[dt[i]._className][dt[i]._index].width - 1))); // which cols
			real_y = std::max(0, std::min(real_y, static_cast<int>(md->_imagsProperty[dt[i]._className][dt[i]._index].height - 1))); // which rows
			int tmp = static_cast<int>(dt[i]._img.ptr<uchar>(real_y)[real_x]); //real_y at first

			delta_x = dt[i]._rotation(0, 0)*pos.end.x + dt[i]._rotation(0, 1)*pos.end.y;
			delta_y = dt[i]._rotation(1, 0)*pos.end.x + dt[i]._rotation(1, 1)*pos.end.y;
			delta_x = dt[i]._scale*delta_x*dt[i]._bbox.width / 2.0;//在框内的坐标
			delta_y = dt[i]._scale*delta_y*dt[i]._bbox.height / 2.0;
			real_x = delta_x + dt[i]._prdshape(_landmark_index, 0);//在整个图片内的坐标
			real_y = delta_y + dt[i]._prdshape(_landmark_index, 1);
			real_x = std::max(0, std::min(real_x, static_cast<int>(md->_imagsProperty[dt[i]._className][dt[i]._index].width - 1))); // which cols
			real_y = std::max(0, std::min(real_y, static_cast<int>(md->_imagsProperty[dt[i]._className][dt[i]._index].height - 1))); // which rows
			dt[i]._pixDiffFeat(j) = tmp - static_cast<int>(dt[i]._img.ptr<uchar>(real_y)[real_x]);
		}
	}
}

CorR RandomForest::Split_Type(const int stage)
{
	float threshold = 1 - 0.1*stage;
	float rdnumber = RandNumberUniform<float>(0.0, 1.0);
	if (threshold <= rdnumber)
		return REGRESSION;
	else
		return CLASSIFICATION;
}

void RandomForest::AssignCScore_Node(Node* nd, std::deque<DT>& p_dt, std::deque<DT>& n_dt)
{
	double numerator = 0.0;
	double denominator = 0.0;
	for (int i = 0; i < p_dt.size(); i++)
		numerator += p_dt[i]._weight;
	for (int i = 0; i < n_dt.size(); i++)
		denominator += n_dt[i]._weight;

	nd->_cscore = numerator / denominator;
}

void RandomForest::getCscore_singleTress(Node* nd, DT& dt)
{
	if (true == nd->_is_leaf)
	{
		dt._score += nd->_cscore; //Accumulating the classification score
		return;
	}
	else
	{
		getCscore_singleTress(nd->_left_child, dt);
		getCscore_singleTress(nd->_right_child, dt);
	}
}