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

	std::deque<DT> p_Dt, n_Dt;
		//填充样本结构（正样本）
	for (int i = 0; i < _pm._n_p; i++)
	{
		DT temp;
		
		temp._className = "POSITIVE";
		temp._weight = 1.0;
		temp._score = 0.0;
		temp._bbox = md->_bbox_origial[i];
		temp._gtshape = md->_gtShape[i];
		temp._lable = 1;
		temp._path = md->_imagsPath["POSITIVE"][i];
		temp._index = i;
		temp._img = cv::imread(temp._path, 0);
		temp._prdshape = ProjectShape(md->_Meanshape.col(0), md->_Meanshape.col(1), temp._bbox); //Initializing the predicted shape with mean shape, it will updated after
		p_Dt.push_back(temp);

	}
		//填充样本结构（负样本）
	for (int i = 0; i < _pm._n_n; i++)
	{
		DT temp = GeNegSamp(md);
		n_Dt.push_back(temp);
	}

	//STEP2. Train
	for (uint16 t = 0; t < _pm._T; t++)//Foreach Stage（Cascade）
	{
		//
		std::cout << "----------Training Stage : " << t <<" ----------"<< std::endl;
		for (uint32 k = 0; k < _pm._K; k++)//Foreach weak classifier (Binary Tree)
		{
			//Computing samples' weight(Not for the first stage) 
			
			if (0 != t)
			{
				for each (DT var in p_Dt)
					AsignWeight(var);
				for each (DT var in n_Dt)
					AsignWeight(var);
			}

			//Select a point, and constructing its random forest
			int pt = k / _pm._L;
			
			//Learn the structure of classification/regression trees CR
			std::cout << "Training weak classifer: " << k << "---pt: " << pt << "----- stage: " << t << std::endl;



			
		}

		//
	}


	//


}


//Generating negative samples randomly 
DT JDA::GeNegSamp(MYDATA* const md)
{

	int rdn = RandNumberUniform<int>(0, _pm._n_n - 1);//Selecting a sample from negative set randomly

	/* code */
	DT result;
	float w = md->_imagsProperty["NEGATIVE"][rdn].width;
	float h = md->_imagsProperty["NEGATIVE"][rdn].height;

	result._bbox.x = RandNumberUniform<float>(0.2, 0.8)*w; //This scope can be varied  
	result._bbox.y = RandNumberUniform<float>(0.2, 0.8)*h;
	result._bbox.width = RandNumberUniform<float>(0.2, 0.5)*w;
	result._bbox.height = RandNumberUniform<float>(0.2, 0.5)*h;
	result._bbox.ctx = result._bbox.x + result._bbox.width / 2;
	result._bbox.cty = result._bbox.y + result._bbox.height / 2;

	result._className = "NEGATIVE";
	result._index = rdn;
	result._weight = 1.0;
	result._score = 0.0;
	result._lable = -1;
	result._path = md->_imagsPath["NEGATIVE"][rdn];
	result._img = cv::imread(result._path, 0);
	result._prdshape = ProjectShape(md->_Meanshape.col(0), md->_Meanshape.col(1), result._bbox); 

	return result;
}

//Function for leaning CR tress
void JDA::LearnCRTrees(const std::deque<DT>& p_dt, const std::deque<DT>& n_dt, const PARAMETERS& pm)
{

}
