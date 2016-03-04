#include "stdafx.h"
#include "JDA.h"



JDA::JDA(){}
JDA::JDA(const PARAMETERS& pm){_pm = pm;}
JDA::~JDA(){}

void JDA::train(MYDATA* const md)
{
	//SETP1. We need checkout positive samples and generate negative samples
	//1.���ڸ������ĳ�ȡ������Ϊ�ڳ���ͼƬ��������ɿ�Ȼ���meanshapeͶӰ��ȥ����Ĵ�С��λ�ò���Ϊ����ͼƬ�����ĳ�����������ֵ
	//2.����������ĿҪά����һ������ _pm._nn, ���������Ļ����Ų��ϵ�ȥ�����ڷ�����ֵ������������٣�Ҫѡ�ú���������������Щ��Ϣ

	std::deque<DT*> p_Dt, n_Dt;
		//��������ṹ����������
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
		//��������ṹ����������
	for (int i = 0; i < _pm._n_n; i++)
	{
		DT* temp = GeNegSamp(md, _pm);
		n_Dt.push_back(temp);
	}

	//STEP2. Train
	std::vector<std::vector<Node*>>cascade;
	for (uint16 t = 0; t < _pm._T; t++)//Foreach Stage��Cascade��
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
