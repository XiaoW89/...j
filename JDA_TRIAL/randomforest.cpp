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
	_samples_neg = -1;
	_samples_pos = -1;
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


	// train Random Forest
	// construct each tree in the forest
	_local_position.clear();
	_local_position.resize(pm._L);
	trees_.clear();
	trees_.resize(pm._L);
	std::set<int>selected_indexes;
	for (int i = 0; i < pm._K; i++)
	{
		//********STEP 1 : Computing samples' weight(Not for the first stage) ********
		if (0 != (_stage+i) )
		{
			for each (DT* var in p_dt)
				AsignWeight(var);
			for each (DT* var in n_dt)
				AsignWeight(var);
		}

		//********STEP 2 : Select a point for regression********
		_landmark_index = i / _trees_num_per_forest; //determine which landmark need to be trained 
		std::cout << "Training weak classifer: " << i << "( pt: " << _landmark_index << "'th,  stage: " << _stage << " )" << std::endl;
		//********STEP 3 : Extract features and regression target********
		if (0 == (i%_trees_num_per_forest)) //同一个特征点对应的所有树都共享一份坐标
		{

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

			std::cout << "get pixel differences" << std::endl;
			GeneratePixelDiff(md, p_dt, _local_position[_landmark_index]);
			GeneratePixelDiff(md, n_dt, _local_position[_landmark_index]);

			selected_indexes.clear();
		}

		//for (int j = 0; j < p_dt.size();j++)
		//{
		//	

		//	for (int tt = 0; tt < _local_features_num;tt++)
		//	{
		//		
		//		DT* dt = p_dt[j];

		//		cv::Mat img = cv::imread(dt->_path);
		//		cv::circle(img, cv::Point2i(dt->_prdshape(_landmark_index, 0), dt->_prdshape(_landmark_index, 1)), 2, cv::Scalar(0, 0, 255));

		//		FeatureLocations fl = _local_position[_landmark_index][tt];
		//		double delta_x = dt->_rotation(0, 0)*fl.start.x + dt->_rotation(0, 1)*fl.start.y;//旋转后的x
		//		double delta_y = dt->_rotation(1, 0)*fl.start.x + dt->_rotation(1, 1)*fl.start.y;//旋转后的y
		//		delta_x = dt->_scale*delta_x*dt->_bbox.width / 2.0;//在框内的坐标
		//		delta_y = dt->_scale*delta_y*dt->_bbox.height / 2.0;
		//		int real_x = delta_x + dt->_prdshape(_landmark_index, 0);//在整个图片内的坐标
		//		int real_y = delta_y + dt->_prdshape(_landmark_index, 1);
		//		real_x = (std::max)(0, (std::min)(real_x, static_cast<int>(md->_imagsProperty[dt->_className][dt->_index].width - 1))); // which cols
		//		real_y = (std::max)(0, (std::min)(real_y, static_cast<int>(md->_imagsProperty[dt->_className][dt->_index].height - 1))); // which rows
		//		int tmp = static_cast<int>(dt->_img.ptr<uchar>(real_y)[real_x]); //real_y at first

		//		cv::circle(img, cv::Point2i(real_x, real_y), 2, cv::Scalar(0, 0, 255));

		//		delta_x = dt->_rotation(0, 0)*fl.end.x + dt->_rotation(0, 1)*fl.end.y;
		//		delta_y = dt->_rotation(1, 0)*fl.end.x + dt->_rotation(1, 1)*fl.end.y;
		//		delta_x = dt->_scale*delta_x*dt->_bbox.width / 2.0;//在框内的坐标
		//		delta_y = dt->_scale*delta_y*dt->_bbox.height / 2.0;
		//		real_x = delta_x + dt->_prdshape(_landmark_index, 0);//在整个图片内的坐标
		//		real_y = delta_y + dt->_prdshape(_landmark_index, 1);
		//		real_x = (std::max)(0, (std::min)(real_x, static_cast<int>(md->_imagsProperty[dt->_className][dt->_index].width - 1))); // which cols
		//		real_y = (std::max)(0, (std::min)(real_y, static_cast<int>(md->_imagsProperty[dt->_className][dt->_index].height - 1))); // which rows
		//		dt->_pixDiffFeat(j) = tmp - static_cast<int>(dt->_img.ptr<uchar>(real_y)[real_x]);

		//		cv::circle(img, cv::Point2i(real_x, real_y), 2, cv::Scalar(0, 0, 255));
		//		rectangle(img, cv::Rect(p_dt[j]->_bbox.x, p_dt[j]->_bbox.y, p_dt[j]->_bbox.width, p_dt[j]->_bbox.height), cv::Scalar(0, 255, 0));
		//		cv::imshow("p", img);
		//		cv::waitKey(0);
		//	}
		//	
		//}
		

		//********STEP 4 : Learn the structure of crtree********
		Node* root = NULL;
		bool falltree = true;  //判断是否是一颗完整的树，是否缺少子树
		std::set<int>selected_indexes_tp;
		int _all_leaf_nodes_tp  = _all_leaf_nodes;
		do 
		{
			falltree = true;
			_all_leaf_nodes = _all_leaf_nodes_tp;
			selected_indexes_tp = selected_indexes;
			root = BuildCRTree(selected_indexes_tp, 0, p_dt, n_dt, falltree);
			if (!falltree)
				std::cerr << "build tree failed: "  << std::endl;
		} while (!falltree);
		
		selected_indexes = selected_indexes_tp;

		//********SETP 5 : Update classification score for each sample********
		std::set<float>score_set;


		for (int j = 0; j < p_dt.size(); j++)
		{
			getCscore_singleTress(root, p_dt[j]);
			score_set.insert(p_dt[j]->_cscore);
		}


		for (int j = 0; j < n_dt.size(); j++)
		{
			getCscore_singleTress(root, n_dt[j]);
			score_set.insert(n_dt[j]->_cscore);
		}

		//********STEP 5 : Determing the bias theta according to a preset precision-recall condition********
		
		std::multiset<ASTANDAR> tvar;

		for (std::set<float>::const_iterator it = score_set.cbegin(); it != score_set.cend(); it++)
		{
			float theta_tp = *it;
			float fp = 0.0; int tp = 0.0;
			for (int t = 0; t < p_dt.size(); t++)
			{
				if (p_dt[t]->_cscore >= theta_tp)
					tp += 1;
			}
			for (int t = 0; t < n_dt.size(); t++)
			{
				if (n_dt[t]->_cscore >= theta_tp)
					fp += 1;
			}
			float TPR = (tp) / (float)(p_dt.size());
			float FPR = (fp ) / (float)(n_dt.size());

			//std::cout << "[ " << TPR << " , " << FPR << " ]"<<std::endl;
			//double def_tpr = 0.0, def_fpr = 0.0;
			//if (0 == _stage)
			//{
			//	def_tpr = 0.95;   def_fpr = 0.95;
			//}
			//else
			//{
			//	def_tpr = 0.9;   def_fpr = 0.8;
			//}

			//if (TPR > def_tpr && FPR<def_fpr)
			//{
			//	ASTANDAR as;
			//	as._FPR = FPR;  as._Theta = theta_tp; as._TPR = TPR;
			//	tvar.insert(as);
			//}

			if (FPR < 0.95)
			{
				ASTANDAR as;
				as._FPR = FPR;  as._Theta = theta_tp; as._TPR = TPR;
				tvar.insert(as);
			}
		}

		/*for (std::multiset<ASTANDAR>::iterator j = tvar.begin(); j != tvar.end(); j++)
		{
		std::cout << (*j)._TPR << "   " << (*j)._FPR << std::endl;
		}*/


		if (tvar.size() == 0)
			std::cerr << "No appropirate Theta !";

		root->as = *tvar.begin();

		std::cout << i << "'th weak classifier with --theta : " << root->as._Theta ;
		std::cout << "  _TPR : " << root->as._TPR;
		std::cout << "  _FPR : " << root->as._FPR << std::endl;

		this->trees_[_landmark_index].push_back(root);

		//********STEP 6 : Removing samples whos classification score less than theta********
		//if ((i + 1) % _trees_num_per_forest != 0) //每构建_trees_num_per_forest个分类器后进行筛选
		//	continue;

		std::deque<DT*>::iterator it = p_dt.begin();
		int rm_neg = 0, rm_pos = 0;

		while (it != p_dt.end())
		{
			if ((*it)->_cscore < root->as._Theta)
			{
				(*it)->_label_preditced = -1;
				delete *it;
				it = p_dt.erase(it);
				rm_pos++;
			}
			else
			{
				(*it)->_label_preditced = 1;
				it++;
			}
				
		}
		it = n_dt.begin();
		while (it != n_dt.end())
		{
			if ((*it)->_cscore < root->as._Theta)
			{
				(*it)->_label_preditced = -1;
				delete *it;
				it = n_dt.erase(it);
				rm_neg++;
			}
			else
			{
				(*it)->_label_preditced = 1;
				it++;
			}
		}

		std::cout << "had removed : " << rm_pos << "pos samples, and " << rm_neg << "neg samples" << std::endl;
		std::cout << "Surplus of pos sample : " << p_dt.size() << std::endl;

		//********STEP 7 : Peform negative sample mining if negative samples are insufficient
		
		std::cout << "Mining negtive sample ......." << std::endl;
		float sampflag = 0.0;
		float total = 0.0;

		while(p_dt.size() - n_dt.size()>0)
		{
			{
				total++;
				//Firstly, we generate a intial negative sample
				DT* temp = GeNegSamp(md, pm);

				//Then, let this sample traverse to current tree
				//Attention, this procedure will update the prdshape of each input dt
				bool passflag = getCscore_wholeTress(cascade, shape_param_set, md, temp);

				//Lastly, put it into the negative sample set if it survived in last step
				if (passflag)
				{
					n_dt.push_back(temp);
					sampflag += 1;
				}
				else
				{
					delete temp;
				}
			}
		}
		std::cout << "pass rate : " << sampflag / total << std::endl<<std::endl;
	}
	return true;
}


Node* RandomForest::BuildCRTree(std::set<int>& selected_ft_indexes, int current_depth, std::deque<DT*>& p_dt,
	std::deque<DT*>& n_dt, bool & falltree)
{
	if (p_dt.size()+ n_dt.size()>0)// the node may not split under some cases
	{ 
		//Decide either CLASSIFICATION or REGRESSION node
		CorR type_split = Split_Type(current_depth);

		//Construct tree iteratively
		Node* node = new Node();
		node->_depth = current_depth;
		node->_samples_neg = n_dt.size();
		node->_samples_pos = p_dt.size();
	
		std::deque<DT*> left_image_pos, right_image_pos,
			left_image_neg, right_image_neg;
		if (current_depth == _tree_depth){ // the node reaches max depth
			node->_is_leaf = true;
			node->_leaf_identity = _all_leaf_nodes;
			_all_leaf_nodes++;
			AssignCScore_Node(node, p_dt, n_dt); //Assign classification score for leaf node
			return node;
		}

		int ret = FindSplitFeature_random(node, selected_ft_indexes, left_image_pos, left_image_neg,
			right_image_pos,right_image_neg, type_split, p_dt, n_dt);

	
		// actually it won't enter the if block, when the random function is good enough
		if (ret == 1) // the current node contain all sample when reaches max variance reduction, it is leaf node
		{
			std::cout <<std::endl<< "WARMING : this node contain don't reach specified depth !" << std::endl;
			/*node->_is_leaf = true;
			node->_leaf_identity = _all_leaf_nodes;
			_all_leaf_nodes++;
			AssignCScore_Node(node, p_dt, n_dt);*/
			falltree = false;
			return NULL;
		}

		node->_left_child = BuildCRTree(selected_ft_indexes, current_depth + 1, left_image_pos, left_image_neg, falltree);
		node->_right_child = BuildCRTree(selected_ft_indexes, current_depth + 1, right_image_pos, right_image_neg, falltree);

		return node;
	}
	else{ // this case is not possible in this data structure
		
		falltree = false;
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
	tp_left_pos.clear(); tp_left_neg.clear(); tp_right_neg.clear(); tp_right_pos.clear();

	std::vector<int> data;


	for (int j = 0; j < _local_features_num; j++)
	{
		if (selected_ft_indexes.find(j) == selected_ft_indexes.end()) //如果j不在容器里面就执行操作
		{
			//I am training regression nodes with positive samples only
			if (REGRESSION == corr) //For Regression Node
			{
				// random generate threshold (****Need be modifed)
				data.clear();
				for (int i = 0; i < n_p; i++)
				{
					data.push_back(p_dt[i]->_pixDiffFeat(j));
				}
				std::sort(data.begin(), data.end());
				data.erase(std::unique(data.begin(), data.end()), data.end());
				if (data.size()<=5)
				{
					std::cerr << "The candidate threshold of pixel feature :" << j << ", is insufficient!!" << std::endl;
					for (int tt = 0; tt < p_dt.size(); tt++)
					{
						std::cout << p_dt[tt]->_pixDiffFeat(j) << "   " << std::endl;
					}
					continue;
				}
				bool right_direction = false;

				for (int z = 1; z < data.size() / 2; z++)
				{
					double var_tmp_mr = regreesion_split_test(p_dt, data[data.size() / 2 + z], j, tp_left_pos, tp_right_pos); //中点右侧的尝试
					double var_tmp_ml = regreesion_split_test(p_dt, data[data.size() / 2 - z], j, tp_left_pos, tp_right_pos); //中点左侧的尝试
					if (var_tmp_mr > var_tmp_ml)
					{
						right_direction = true;
						break;
					}
					else if (var_tmp_mr < var_tmp_ml)
					{
						right_direction = false;
						break;
					}
					else
						continue;
				}

				int it = 0;
				while (it < data.size()/2-1)
				{
					int tmp_threshold = 0;
					if (right_direction)
					{
						tmp_threshold = data[data.size()/2+it];
					}
					else
					{
						tmp_threshold = data[data.size() / 2 - it];
					}
						
					double var_red = regreesion_split_test(p_dt, tmp_threshold, j, tp_left_pos, tp_right_pos);
					if (var_red > var) //这里用的是最小均方差LSD，主要是希望类内方差小
					{
						var = var_red;
						threshold = tmp_threshold;
						feature_index = j;
						left_neg = right_neg = n_dt; //原封不动的送入下一层
						left_pos = tp_left_pos;
						right_pos = tp_right_pos;
					}
					it+=4;
				}
			}
			else //For classification node , notice that the weight of each sample must taked into account when performing split test
			{
				// random generate threshold (****Need be modifed)
				data.clear();
				for (int i = 0; i < p_dt.size(); i++){
					data.push_back(p_dt[i]->_pixDiffFeat(j));
				}
				for (int i = 0; i < n_dt.size(); i++){
					data.push_back(n_dt[i]->_pixDiffFeat(j));
				}
				std::sort(data.begin(), data.end());
				data.erase(std::unique(data.begin(), data.end()), data.end());
				if (data.size() <= 5)
				{
					std::cerr << "The candidate threshold of pixel feature :" << j << ", is insufficient!!" << std::endl;
					for (int tt = 0; tt < p_dt.size();tt++)
					{
						std::cout << p_dt[tt]->_pixDiffFeat(j) << "   " << std::endl;
					}
					for (int tt = 0; tt < n_dt.size(); tt++)
					{
						std::cout << n_dt[tt]->_pixDiffFeat(j) << "   " << std::endl;
					}
					continue;
				}
				bool right_direction = false;

				for (int z = 1; z < data.size() / 2 - 2; z++)
				{
					double var_tmp_mr = classification_split_test(p_dt, n_dt, data[data.size() / 2 + z], j, tp_left_pos, tp_right_pos, tp_right_neg, tp_left_neg); //中点右侧的尝试
					double var_tmp_ml = classification_split_test(p_dt, n_dt, data[data.size() / 2 - z], j, tp_left_pos, tp_right_pos, tp_right_neg, tp_left_neg); //中点左侧的尝试
					if (var_tmp_mr > var_tmp_ml)
					{
						right_direction = true;
						break;
					}
					else if (var_tmp_mr < var_tmp_ml)
					{
						right_direction = false;
						break;
					}
					else
						continue;
				}

				int it = 0;
				while (it < data.size() / 2 - 1)
				{
					int tmp_threshold = 0;
					if (right_direction)
					{
						tmp_threshold = data[data.size() / 2 + it];
					}
					else
					{
						tmp_threshold = data[data.size() / 2 - it];
					}
					double inform_gain = classification_split_test(p_dt, n_dt, tmp_threshold, j, tp_left_pos, tp_right_pos, tp_right_neg, tp_left_neg);
					if (inform_gain > entrp)
					{
						entrp = inform_gain;
						threshold = tmp_threshold;
						feature_index = j;
						left_neg = tp_left_neg;
						left_pos = tp_left_pos;
						right_neg = tp_right_neg;
						right_pos = tp_right_pos;
					}
					it += 4;
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
int RandomForest::FindSplitFeature_random(Node* node, std::set<int>& selected_ft_indexes, std::deque<DT*>& left_pos,
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
	tp_left_pos.clear(); tp_left_neg.clear(); tp_right_neg.clear(); tp_right_pos.clear();

	std::vector<int> data;

	int tflag = 0; 

	while (tflag <= _local_features_num)
	{
		for (int j = 0; j < _local_features_num; j++)
		{
			if (selected_ft_indexes.find(j) == selected_ft_indexes.end()) //如果j不在容器里面就执行操作
			{
				//I am training regression nodes with positive samples only
				if (REGRESSION == corr) //For Regression Node
				{
					// random generate threshold (****Need be modifed)
					data.clear();

					for (int i = 0; i < n_p; i++)
					{
						data.push_back(p_dt[i]->_pixDiffFeat(j));
					}

					if (data.size() == 0)
						return 1;
					std::sort(data.begin(), data.end());
					data.erase(std::unique(data.begin(), data.end()), data.end());

					int tmp_threshold = data[RandNumberUniform<int>(0, data.size() - 1)];

					double var_red = regreesion_split_test(p_dt, tmp_threshold, j, tp_left_pos, tp_right_pos);
					if (var_red > var) //这里用的是最小均方差LSD，主要是希望类内方差小
					{
						var = var_red;
						threshold = tmp_threshold;
						feature_index = j;
						left_neg = right_neg = n_dt; //原封不动的送入下一层
						left_pos = tp_left_pos;
						right_pos = tp_right_pos;
					}
				}
				else //For classification node , notice that the weight of each sample must taked into account when performing split test
				{
					// random generate threshold (****Need be modifed)
					data.clear();
					for (int i = 0; i < p_dt.size(); i++){
						data.push_back(p_dt[i]->_pixDiffFeat(j));
					}
					for (int i = 0; i < n_dt.size(); i++){
						data.push_back(n_dt[i]->_pixDiffFeat(j));
					}
					std::sort(data.begin(), data.end());
					data.erase(std::unique(data.begin(), data.end()), data.end());

					if (data.size() == 0)
						return 1;

					int tmp_threshold = data[RandNumberUniform<int>(0, data.size() - 1)];

					double inform_gain = classification_split_test(p_dt, n_dt, tmp_threshold, j, tp_left_pos, tp_right_pos, tp_right_neg, tp_left_neg);
					if (inform_gain > entrp)
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
				//std::cout << "Waring: this node contain all the samples" << std::endl;
				if (tflag < _local_features_num)
				{
					tflag++;
					continue;
				}
				else
				{
					node->_is_leaf = true; // the node can contain all the samples
					tflag = true;
					return 1;
				}
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
		else
		{
			if (tflag < _local_features_num)
			{
				tflag++;
				continue;
			}
			else
				return -1;
		}
	}
}

RandomForest::RandomForest(const PARAMETERS& param, const int stage)
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

void RandomForest::GeneratePixelDiff(MYDATA* const md, DT* dt, const std::vector<FeatureLocations>& fl_set)
{
		//getSimilarityTransform(ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]),mean_shape_, rotation, scale);
	for (int j = 0; j < _local_features_num; j++)
	{
		FeatureLocations fl = fl_set[j];
		double delta_x = dt->_rotation(0, 0)*fl.start.x + dt->_rotation(0, 1)*fl.start.y;//旋转后的x
		double delta_y = dt->_rotation(1, 0)*fl.start.x + dt->_rotation(1, 1)*fl.start.y;//旋转后的y
		delta_x = dt->_scale*delta_x*dt->_bbox.width / 2.0;//在框内的坐标
		delta_y = dt->_scale*delta_y*dt->_bbox.height / 2.0;
		int real_x = delta_x + dt->_prdshape(_landmark_index, 0);//在整个图片内的坐标
		int real_y = delta_y + dt->_prdshape(_landmark_index, 1);
		real_x = (std::max)(0, (std::min)(real_x, static_cast<int>(md->_imagsProperty[dt->_className][dt->_index].width - 1))); // which cols
		real_y = (std::max)(0, (std::min)(real_y, static_cast<int>(md->_imagsProperty[dt->_className][dt->_index].height - 1))); // which rows

		int tmp = 0;
		if ("POSITIVE" == dt->_className)
		{
			tmp = static_cast<int>(md->pos_images[dt->_index].ptr<uchar>(real_y)[real_x]); //real_y at first
		}
		else
		{
			tmp = static_cast<int>(md->neg_images[dt->_index].ptr<uchar>(real_y)[real_x]); //real_y at first
		}
		
		delta_x = dt->_rotation(0, 0)*fl.end.x + dt->_rotation(0, 1)*fl.end.y;
		delta_y = dt->_rotation(1, 0)*fl.end.x + dt->_rotation(1, 1)*fl.end.y;
		delta_x = dt->_scale*delta_x*dt->_bbox.width / 2.0;//在框内的坐标
		delta_y = dt->_scale*delta_y*dt->_bbox.height / 2.0;
		real_x = delta_x + dt->_prdshape(_landmark_index, 0);//在整个图片内的坐标
		real_y = delta_y + dt->_prdshape(_landmark_index, 1);
		
		real_x = (std::max)(0, (std::min)(real_x, static_cast<int>(md->_imagsProperty[dt->_className][dt->_index].width - 1))); // which cols
		real_y = (std::max)(0, (std::min)(real_y, static_cast<int>(md->_imagsProperty[dt->_className][dt->_index].height - 1))); // which rows
		if ("POSITIVE" == dt->_className)
			dt->_pixDiffFeat(j) = tmp - static_cast<int>(md->pos_images[dt->_index].ptr<uchar>(real_y)[real_x]);
		else
			dt->_pixDiffFeat(j) = tmp - static_cast<int>(md->neg_images[dt->_index].ptr<uchar>(real_y)[real_x]);
	}
}

void RandomForest::GeneratePixelDiff(MYDATA* const md, std::deque<DT*>& dt, const std::vector<FeatureLocations>& fl_set)
{
	for (int i = 0; i < dt.size(); i++)
		GeneratePixelDiff(md, dt[i], fl_set);
}

CorR RandomForest::Split_Type(const int depth)
{
	float threshold = 1 - 0.1*(_stage+1);
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

	nd->_cscore = log(numerator / denominator)/2;
}

void RandomForest::getCscore_singleTress(const Node* nd,  DT* dt)
{
	if (true == nd->_is_leaf)
	{
		dt->_cscore += nd->_cscore; //Accumulating the classification score
		return;
	}
	else
	{
		if (dt->_pixDiffFeat(nd->ft_index) < nd->_threshold)
			getCscore_singleTress(nd->_left_child, dt);
		else
			getCscore_singleTress(nd->_right_child, dt);
	}
}

bool RandomForest::getCscore_wholeTress(std::vector<RandomForest>&cascade,
	const std::vector<cv::Mat_<float>>&_shape_param_set, MYDATA* md, DT* dt)
{

	if (0 == _stage)
	{
		calcRot_target(md->_Meanshape, dt);
		for (int i = 0; i < _landmark_index+1; i++) //加1是因为_landmark_index是从0开始的，如果不加1则这个循环在第一个阶段的前几个弱分类器无法执行
		{
			GeneratePixelDiff(md, dt, _local_position[i]);
			for (int j = 0; j < trees_[i].size(); j++)
			{
				if ((i+j)!= 0) //第一次送进来的时候是不需要进行权重更新的，统统都是1，后续的需要更新
					AsignWeight(dt);
				getCscore_singleTress(trees_[i][j], dt);
				//if (j == _trees_num_per_forest-1)
					if (dt->_cscore < trees_[i][j]->as._Theta)
						return false;
			}
		}
	}
	else
	{
		for (int t = 0; t < cascade.size(); t++)
		{
			calcRot_target(md->_Meanshape, dt);
			for (int i = 0; i < cascade[t]._landmark_index + 1; i++) //foreach cascade
			{
				GeneratePixelDiff(md, dt, cascade[t]._local_position[i]);
				for (int j = 0; j < cascade[t].trees_[i].size(); j++)
				{
					if ((t + i+ j)!=0)
						AsignWeight(dt);
					getCscore_singleTress(cascade[t].trees_[i][j], dt);
					//if (j == _trees_num_per_forest - 1)
						if (dt->_cscore < cascade[t].trees_[i][j]->as._Theta)
							return  false;
				}
			}
			GetGlobalLBF(md, cascade[t], dt);
			UpdateShape(_shape_param_set[t], dt);
		}
	}
	return true;
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

double RandomForest::regreesion_split_test(const std::deque<DT*>&p_dt, const double tmp_threshold, const int ft_index,  std::deque<DT*>&tp_left_pos, 
	std::deque<DT*>&tp_right_pos)
{
	tp_left_pos.clear();  tp_right_pos.clear();
	double var_lc = 0.0, var_rc = 0.0, var_red = 0.0;
	double Ex_2_lc = 0.0, Ex_lc = 0.0, Ey_2_lc = 0.0, Ey_lc = 0.0;
	double Ex_2_rc = 0.0, Ex_rc = 0.0, Ey_2_rc = 0.0, Ey_rc = 0.0;

	for (int i = 0; i < p_dt.size(); i++)
	{
		if ((p_dt[i]->_pixDiffFeat(ft_index) < tmp_threshold))
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

	return var_red;
}

double RandomForest::classification_split_test(const std::deque<DT*>&p_dt, const std::deque<DT*>&n_dt, const double tmp_threshold, const int ft_index, std::deque<DT*>&tp_left_pos,
	std::deque<DT*>&tp_right_pos, std::deque<DT*>&tp_right_neg, std::deque<DT*>&tp_left_neg)
{
	tp_left_pos.clear(); tp_left_neg.clear(); tp_right_neg.clear(); tp_right_pos.clear();
	float weight_n_n_l = 0, weight_n_n_r = 0, weight_n_p_l = 0, weight_n_p_r = 0;
	for (int i = 0; i < p_dt.size(); i++)
	{
		if ((p_dt[i]->_pixDiffFeat(ft_index) < tmp_threshold))
		{
			tp_left_pos.push_back(p_dt[i]);
			weight_n_p_l += p_dt[i]->_weight;
		}
		else
		{
			tp_right_pos.push_back(p_dt[i]);
			weight_n_p_r += p_dt[i]->_weight;
		}
	}
	for (int i = 0; i <n_dt.size() ; i++)
	{
		if ((n_dt[i]->_pixDiffFeat(ft_index) < tmp_threshold))
		{
			tp_left_neg.push_back(n_dt[i]);
			weight_n_n_l += n_dt[i]->_weight;
		}
		else
		{
			tp_right_neg.push_back(n_dt[i]);
			weight_n_n_r += n_dt[i]->_weight;
		}
	}

	float weight_n_n = 0, weight_n_p = 0;
	for (int i = 0; i < p_dt.size(); i++)
		weight_n_p += p_dt[i]->_weight;
	for (int i = 0; i < n_dt.size(); i++)
		weight_n_n += n_dt[i]->_weight;


	double entrp_root = -(weight_n_n / (weight_n_n + weight_n_p))*log((weight_n_n / (weight_n_n + weight_n_p))) - (weight_n_p / (weight_n_n + weight_n_p))*log((weight_n_p / (weight_n_n + weight_n_p)));
	double entrp_l = -(weight_n_n_l / (weight_n_n_l + weight_n_p_l))*log((weight_n_n_l / (weight_n_n_l + weight_n_p_l))) - (weight_n_p_l / (weight_n_n_l + weight_n_p_l))*log((weight_n_p_l / (weight_n_n_l + weight_n_p_l)));
	double entrp_r = -(weight_n_n_r / (weight_n_n_r + weight_n_p_r))*log((weight_n_n_r / (weight_n_n_r + weight_n_p_r))) - (weight_n_p_r / (weight_n_n_r + weight_n_p_r))*log((weight_n_p_r / (weight_n_n_r + weight_n_p_r)));

	double inform_gain = entrp_root - (weight_n_n_l + weight_n_p_l)*entrp_l / (weight_n_n + weight_n_p) - (weight_n_n_r + weight_n_p_r)*entrp_r / (weight_n_n + weight_n_p);

	return inform_gain;
}

void getlocallbf(const Node* nd, DT* dt)
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

void GetGlobalLBF(MYDATA* md, RandomForest& rf, DT* dt)
{
	std::deque<DT*> dt_temp;
	dt_temp.push_back(dt);
	dt->_LBF = cv::Mat::zeros(1, rf._trees_num_per_forest*(rf._landmark_index + 1), CV_32FC1);

	for (int i = 0; i < rf.trees_.size(); i++)
	{
		if (0 == (i%rf._trees_num_per_forest))
			rf.GeneratePixelDiff(md, dt_temp, rf._local_position[i / rf._trees_num_per_forest]);
		getlocallbf(rf.trees_[i / rf._trees_num_per_forest][i%rf._trees_num_per_forest], dt);
	}
}

