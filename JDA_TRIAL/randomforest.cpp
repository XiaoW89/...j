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

bool RandomForest::TrainForest(MYDATA* const md,  const PARAMETERS& pm, 
	const std::vector<cv::Mat_<float>>&shape_param_set, std::deque<DT*>& p_dt,
	std::deque<DT*>& n_dt, std::vector<RandomForest>&cascade)
{
	//注意， 在训练每个rf得过程里， 有些变量得存储形式是以二维vector得形式保存得如trees_, _local_position, 
	//这样做的原因在于我是把整个rf划分成pm._L个小森林， _L是特征点的个数， 即每个特征点对应了_trees_num_per_forest
	//颗树，同一个对应的所有树会共享一部分信息，如_local_position,  因而以这种形式保存结果更方便，减少计算

	cv::waitKey(1000);

	int n_sample_neg = n_dt.size(); //a backups for sample size
	int n_sample_pos = p_dt.size(); //

	// train Random Forest
	// construct each tree in the forest
	_local_position.clear();
	_local_position.resize(pm._L);
	trees_.clear();
	trees_.resize(pm._L); //pre-allocate 
	

	for (int i = 0; i < pm._K; i++)
	{
		//********STEP 1 : Computing samples' weight(Not for the first stage) ********
		if (0 != _stage)
		{
			for each (DT* var in p_dt)
				AsignWeight(var);
			for each (DT* var in n_dt)
				AsignWeight(var);
		}

		//********STEP 2 : Select a point for regression********
		_landmark_index = i / _trees_num_per_forest; //determine which landmark need to be trained 

		//********STEP 3 : Extract features and regression target********
		if (0 == (i%_trees_num_per_forest)) //同一个特征点对应的所有树都共享一份坐标
		{
			std::cout << "Training weak classifer: " << i << "( pt: " << _landmark_index << "'th,  stage: " << _stage <<" )"<< std::endl;
			for (int j = 0; j < _local_features_num; j++)
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

				_local_position[_landmark_index].push_back(FeatureLocations(a, b));
			}

		}

			//Extract pixel difference feature
		std::cout << "get pixel differences" << std::endl;
		GeneratePixelDiff(md, p_dt, _local_position[_landmark_index]);
		GeneratePixelDiff(md, n_dt, _local_position[_landmark_index]);


		//********STEP 4 : Learn the structure of crtree********
		std::set<int> selected_indexes; //used for storing features' index when perform node-split 
		Node* root = BuildCRTree(selected_indexes,  0, p_dt, n_dt);
		

		//********SETP 5 : Update classification score for each sample********
		std::set<float>score_set;
		for (int j = 0; j < p_dt.size(); j++)
		{
			getCscore_singleTress(root, p_dt[j]);
			score_set.insert(p_dt[j]->_score);
		}
		for (int j = 0; j < n_dt.size(); j++)
		{
			getCscore_singleTress(root, n_dt[j]);
			score_set.insert(n_dt[j]->_score);
		}

		//********STEP 5 : Determing the bias theta according to a preset precision-recall condition********
		
		std::multiset<ASTANDAR> tvar;

		for (std::set<float>::const_iterator it = score_set.cbegin(); it != score_set.cend(); it++)
		{
			float theta_tp = *it;
			float fp = 0.0; int tp = 0.0;
			for (int t = 0; t < p_dt.size(); t++)
			{
				if (p_dt[t]->_score >= theta_tp)
					tp += 1;
			}
			for (int t = 0; t < n_dt.size(); t++)
			{
				if (n_dt[t]->_score >= theta_tp)
					fp += 1;
			}
			float TPR = (tp) / (float)(p_dt.size());
			float FPR = (fp ) / (float)(n_dt.size());

			std::cout << "[ " << TPR << " , " << FPR << " ]"<<std::endl;
			if (TPR > 0.8 && FPR<0.6)
			{
				ASTANDAR as;
				as._FPR = FPR;  as._Theta = theta_tp; as._TPR = TPR;
				tvar.insert(as);
			}
		}

		root->as = *tvar.begin();

		std::cout << i << "'th weak classifier with --theta : " << root->as._Theta << std::endl;
		std::cout << "_TPR : " << root->as._TPR << std::endl;
		std::cout << "_FPR : " << root->as._FPR << std::endl;

		this->trees_[_landmark_index].push_back(root);

		//********STEP 6 : Removing samples whos classification score less than theta********
		std::deque<DT*>::iterator it = p_dt.begin();
		int rm_neg = 0, rm_pos = 0;
		while (it != p_dt.end())
		{
			if ((*it)->_score < root->as._Theta)
			{
				it = p_dt.erase(it);
				rm_pos++;
			}
			else
				it++;
		}
		it = n_dt.begin();
		while (it != n_dt.end())
		{
			if ((*it)->_score < root->as._Theta)
			{
				it = n_dt.erase(it);
				rm_neg++;
			}
			else
				it++;
		}

		std::cout << "had removed : " << rm_pos << "pos samples, and " << rm_neg << "neg samples" << std::endl;
		std::cout << "Surplus of pos sample : " << p_dt.size() - rm_pos << std::endl;
		//********STEP 7 : Peform negative sample mining if negative samples are insufficient
		int sampflag = 1;
		while ((n_sample_neg - n_dt.size())>0)
		{
			//Firstly, we generate a intial negative sample
			DT* temp = GeNegSamp(md, pm);

			//Then, let this sample traverse to current tree
			//Attention, this procedure will update the prdshape of each input dt
			getCscore_wholeTress(cascade, shape_param_set, md, temp);

			//Lastly, put it into the negative sample set if it survived in last step
			if (temp->_score >= root->_cscore)
			{
				n_dt.push_back(temp);
				std::cout << "Mining the " << sampflag << "'th negtive sample successfully!" << std::endl;
				sampflag += 1;
			}
		}
	}
	return true;
}


Node* RandomForest::BuildCRTree(std::set<int>& selected_ft_indexes, int current_depth, std::deque<DT*>& p_dt,
	std::deque<DT*>& n_dt)
{
	if ((p_dt.size() > 0) & (n_dt.size() > 0))// the node may not split under some cases
	{ 
		//Decide either CLASSIFICATION or REGRESSION node
		CorR type_split = Split_Type(_stage);

		//Construct tree iteratively
		Node* node = new Node();
		node->_depth = current_depth;
		node->_samples = n_dt.size() + p_dt.size();
	
		std::deque<DT*> left_image_pos, right_image_pos,
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


int RandomForest::FindSplitFeature(Node* node, std::set<int>& selected_ft_indexes, std::deque<DT*>& left_pos,
	std::deque<DT*>& left_neg, std::deque<DT*>& right_pos, std::deque<DT*>& right_neg,
	CorR corr, const std::deque<DT*>& p_dt, const std::deque<DT*>& n_dt)
{
	int threshold; 
	double var = -DBL_MAX;
	double entrp = -DBL_MAX;
	int feature_index = -1;

	float n_p = p_dt.size();
	float n_n = n_dt.size();

	std::deque<DT*> tp_left_pos, tp_left_neg, tp_right_pos, tp_right_neg;
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
					data.push_back(p_dt[i]->_pixDiffFeat(j));
				}
				std::sort(data.begin(), data.end());
				int tmp_index = floor((int)(n_p*(0.5 + 0.9*(RandNumberUniform<float>(0.0, 1.0) - 0.5))));
				int tmp_threshold = data[tmp_index];
				for (int i = 0; i < n_p; i++)
				{
					if ((p_dt[i]->_pixDiffFeat(j) < tmp_threshold))
					{
						tp_left_pos.push_back(p_dt[i]);
						// do with regression target
						double value = p_dt[i]->_regressionTarget(_landmark_index, 0);
						Ex_2_lc += pow(value, 2);//左集合中x^2的期望
						Ex_lc += value; //左集合中x的期望
						value = p_dt[i]->_regressionTarget(_landmark_index, 1);
						Ey_2_lc += pow(value, 2);
						Ey_lc += value;
					}
					else
					{
						tp_right_pos.push_back(p_dt[i]);
						double value = p_dt[i]->_regressionTarget(_landmark_index, 0);
						Ex_2_rc += pow(value, 2);
						Ex_rc += value;
						value = p_dt[i]->_regressionTarget(_landmark_index, 1);
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
					data.push_back(p_dt[i]->_pixDiffFeat(j));
				}
				for (int i = 0; i < n_dt.size(); i++){
					data.push_back(n_dt[i]->_pixDiffFeat(j));
				}

				std::sort(data.begin(), data.end());
				int tmp_index = floor((int)(data.size()*(0.5 + 0.9*(RandNumberUniform<float>(0.0, 1.0) - 0.5))));
				int tmp_threshold = data[tmp_index];

				for (int i = 0; i < n_p; i++)
				{
					if ((p_dt[i]->_pixDiffFeat(j) < tmp_threshold)){ tp_left_pos.push_back(p_dt[i]); }
					else{ tp_right_pos.push_back(p_dt[i]); }
				}
				for (int i = 0; i < n_n; i++)
				{
					if ((n_dt[i]->_pixDiffFeat(j) < tmp_threshold)){ tp_left_neg.push_back(n_dt[i]); }
					else{ tp_right_neg.push_back(n_dt[i]); }
				}

				float n_n_l = tp_left_neg.size();
				float n_n_r = tp_right_neg.size();
				float n_p_l = tp_left_pos.size();
				float n_p_r = tp_right_pos.size();

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
		node->ft_index = feature_index;
		node->_feature_locations = _local_position[_landmark_index][feature_index];
		selected_ft_indexes.insert(feature_index);
		if (REGRESSION == corr) 
			node->_cor = REGRESSION;
		else 
			node->_cor = CLASSIFICATION;

		return 0;
	}
	return -1;
}

RandomForest::RandomForest(PARAMETERS& param, int stage)
{
	_stage = stage;
	_local_features_num = param._n_splitFeatures;
	_landmark_index = -1;
	_tree_depth = param._n_deepth;
	_trees_num_per_forest = param._n_childTress;
	_local_radius = param._radius[_stage];
	_all_leaf_nodes = 0;
	//mean_shape_ = param.mean_shape_;
}

RandomForest::RandomForest(){
	
}

void RandomForest::GeneratePixelDiff(MYDATA* const md, std::deque<DT*>& dt, const std::vector<FeatureLocations>& fl_set)
{
	for (int i = 0; i < dt.size(); i++)
	{
		//getSimilarityTransform(ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]),mean_shape_, rotation, scale);

		for (int j = 0; j < _local_features_num; j++)
		{
			FeatureLocations fl = fl_set[j];
			double delta_x = dt[i]->_rotation(0, 0)*fl.start.x + dt[i]->_rotation(0, 1)*fl.start.y;//旋转后的x
			double delta_y = dt[i]->_rotation(1, 0)*fl.start.x + dt[i]->_rotation(1, 1)*fl.start.y;//旋转后的y
			delta_x = dt[i]->_scale*delta_x*dt[i]->_bbox.width / 2.0;//在框内的坐标
			delta_y = dt[i]->_scale*delta_y*dt[i]->_bbox.height / 2.0;
			int real_x = delta_x + dt[i]->_prdshape(_landmark_index, 0);//在整个图片内的坐标
			int real_y = delta_y + dt[i]->_prdshape(_landmark_index, 1);
			real_x = std::max(0, std::min(real_x, static_cast<int>(md->_imagsProperty[dt[i]->_className][dt[i]->_index].width - 1))); // which cols
			real_y = std::max(0, std::min(real_y, static_cast<int>(md->_imagsProperty[dt[i]->_className][dt[i]->_index].height - 1))); // which rows
			int tmp = static_cast<int>(dt[i]->_img.ptr<uchar>(real_y)[real_x]); //real_y at first

			delta_x = dt[i]->_rotation(0, 0)*fl.end.x + dt[i]->_rotation(0, 1)*fl.end.y;
			delta_y = dt[i]->_rotation(1, 0)*fl.end.x + dt[i]->_rotation(1, 1)*fl.end.y;
			delta_x = dt[i]->_scale*delta_x*dt[i]->_bbox.width / 2.0;//在框内的坐标
			delta_y = dt[i]->_scale*delta_y*dt[i]->_bbox.height / 2.0;
			real_x = delta_x + dt[i]->_prdshape(_landmark_index, 0);//在整个图片内的坐标
			real_y = delta_y + dt[i]->_prdshape(_landmark_index, 1);
			real_x = std::max(0, std::min(real_x, static_cast<int>(md->_imagsProperty[dt[i]->_className][dt[i]->_index].width - 1))); // which cols
			real_y = std::max(0, std::min(real_y, static_cast<int>(md->_imagsProperty[dt[i]->_className][dt[i]->_index].height - 1))); // which rows
			dt[i]->_pixDiffFeat(j) = tmp - static_cast<int>(dt[i]->_img.ptr<uchar>(real_y)[real_x]);
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

void RandomForest::AssignCScore_Node(Node* nd, std::deque<DT*>& p_dt, std::deque<DT*>& n_dt)
{
	double numerator = 0.0;
	double denominator = 0.0;
	for (int i = 0; i < p_dt.size(); i++)
		numerator += p_dt[i]->_weight;
	for (int i = 0; i < n_dt.size(); i++)
		denominator += n_dt[i]->_weight;

	nd->_cscore = numerator / denominator;
}

void RandomForest::getCscore_singleTress(const Node* nd,  DT* dt)
{
	if (true == nd->_is_leaf)
	{
		dt->_score += nd->_cscore; //Accumulating the classification score
		return;
	}
	else
	{
		//Extract pixel difference feature
		if (dt->_pixDiffFeat(nd->ft_index) < nd->_threshold)
			getCscore_singleTress(nd->_left_child, dt);
		else
			getCscore_singleTress(nd->_right_child, dt);
	}
}

void RandomForest::getCscore_wholeTress(std::vector<RandomForest>&cascade,
	const std::vector<cv::Mat_<float>>&_shape_param_set, MYDATA* md, DT* dt)
{
	for (int t = 0; t < cascade.size(); t++)
	{
		if (0 == t)
		{
			calcRot_target(md->_Meanshape, dt);
			std::deque<DT*> dt_temp;
			dt_temp.push_back(dt);
			for (int i = 0; i < cascade.size(); i++) //foreach cascade
			{
				for (int j = 0; j < cascade[i].trees_.size(); j++)
				{
					if (0 == (j%_trees_num_per_forest))
						GeneratePixelDiff(md, dt_temp, cascade[i]._local_position[j / _trees_num_per_forest]);
					getCscore_singleTress(cascade[i].trees_[j / _trees_num_per_forest][j%_trees_num_per_forest], dt);
				}
			}
		}
		else
		{
			calcRot_target(md->_Meanshape, dt);
			GetGlobalLBF(md, cascade[t], dt);
			UpdateShape(_shape_param_set[t - 1], dt);
			std::deque<DT*> dt_temp;
			dt_temp.push_back(dt);
			for (int i = 0; i < cascade.size(); i++) //foreach cascade
			{
				for (int j = 0; j < cascade[i].trees_.size(); j++)
				{
					if (0 == (j%_trees_num_per_forest))
						GeneratePixelDiff(md, dt_temp, cascade[i]._local_position[j / _trees_num_per_forest]);
					getCscore_singleTress(cascade[i].trees_[j / _trees_num_per_forest][j%_trees_num_per_forest], dt);
				}
			}
		}
	}
}

//int RandomForest::GetLeafIndex_singleTress(Node* nd, DT* dt)
//{
//	if (true == nd->_is_leaf)
//	{
//		int id = nd->_leaf_identity; //Accumulating the classification score
//		return id;
//	}
//	else
//	{
//		GetLeafIndex_singleTress(nd->_left_child, dt);
//		GetLeafIndex_singleTress(nd->_right_child, dt);
//	}
//}

void RandomForest::getlocallbf(const Node* nd, DT* dt)
{
	static int count = 0;

	if (true == nd->_is_leaf)
	{
		dt->_LBF(nd->_leaf_identity) = 1;  //Accumulating the classification score
		return;
	}
	else
	{
		//Extract pixel difference feature
		if (dt->_pixDiffFeat(nd->ft_index) < nd->_threshold)
			getlocallbf(nd->_left_child, dt);
		else
			getlocallbf(nd->_right_child, dt);
	}
}

void RandomForest::GetGlobalLBF(MYDATA* md, DT* dt)
{
	std::deque<DT*> dt_temp;
	dt_temp.push_back(dt);
	dt->_LBF = cv::Mat::zeros(1, _trees_num_per_forest*(_landmark_index + 1), CV_32FC1);

	for (int i = 0; i < trees_.size(); i++)
	{
		if (0 == (i%_trees_num_per_forest))
			GeneratePixelDiff(md, dt_temp, _local_position[i / _trees_num_per_forest]);
		getlocallbf(trees_[i / _trees_num_per_forest][i%_trees_num_per_forest], dt);
	}
}

void RandomForest::UpdateShape(const cv::Mat_<float>& weights, DT* dt)
{
	cv::Mat_<float> temp;
	temp = dt->_LBF*weights(cv::Rect(0, 0, weights.cols, weights.rows / 2)).t();
	dt->_prdshape.col(0) += temp.t();
	temp = dt->_LBF*weights(cv::Rect(0, weights.rows / 2, weights.cols, weights.rows / 2)).t();
	dt->_prdshape.col(1) += temp.t();

	cv::Mat_<double> rot;
	cv::transpose(dt->_rotation, rot);
	dt->_prdshape = dt->_scale * dt->_prdshape * rot;