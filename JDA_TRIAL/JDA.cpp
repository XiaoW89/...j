#include "stdafx.h"
#include "JDA.h"



JDA::JDA(){}
JDA::JDA(const PARAMETERS& pm){_pm = pm;}
JDA::~JDA(){}

void JDA::train(MYDATA* const md)
{
	//SETP1. We need checkout positive samples and generate negative samples
	//1.关于负样本的抽取，方法为在场景图片中随机生成框，然后把meanshape投影进去，框的大小和位置参数为场景图片长宽的某个比例的随机值
	//2.负样本的数目要维持在一个常量 _pm._nn, 而正样本的话随着不断的去掉低于分类阈值的样本而会减少，要选用合理的容器来存放这些信息

	std::deque<DT*> p_Dt, n_Dt;
		//填充样本结构（正样本）
	for (int i = 0; i < _pm._n_p; i++)
	{
		DT* temp = new DT;
		
		temp->_className = "POSITIVE";
		temp->_weight = 1.0;
		temp->_score = 0.0;
		temp->_bbox = md->_bbox_origial[i];
		temp->_gtshape = md->_gtShape[i];
		temp->_lable = 1;
		temp->_path = md->_imagsPath["POSITIVE"][i];
		temp->_index = i;
		temp->_img = cv::imread(temp->_path, 0);
		temp->_prdshape = ProjectShape(md->_Meanshape.col(0), md->_Meanshape.col(1), temp->_bbox); //Initializing the predicted shape with mean shape, it will updated after
		p_Dt.push_back(temp);

	}
		//填充样本结构（负样本）
	for (int i = 0; i < _pm._n_n; i++)
	{
		DT* temp = GeNegSamp(md, _pm);
		n_Dt.push_back(temp);
	}

	//STEP2. Train
	std::vector<std::vector<Node*>>cascade;
	for (uint16 t = 0; t < _pm._T; t++)//Foreach Stage（Cascade）
	{
		//********STEP 1 : Learn each stage********
		RandomForest rf(_pm, t);
		rf.TrainForest(md, _pm, p_Dt, n_Dt, cascade);

		//********STEP 2 : Learn the shape increments of all leaves (global increments learning)********



		//********STEP 3 : Update shape for all positive samples (include all false positive samples)********



	}

	//


}


//Generating negative samples randomly 


//Function for leaning CR tress
void JDA::LearnCRTrees(const std::deque<DT>& p_dt, const std::deque<DT>& n_dt, const PARAMETERS& pm)
{

}
