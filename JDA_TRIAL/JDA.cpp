#include "stdafx.h"
#include "JDA.h"


JDA::JDA(){}
JDA::JDA(const _Parameters& pm){_pm = pm;}
JDA::~JDA(){}

void JDA::train(const _MyData* md)
{
	//We need checkout positive samples and generate negative samples
	//1.关于负样本的抽取，方法为在场景图片中随机生成框，然后把meanshape投影进去，框的大小和位置参数为场景图片长宽的某个比例的随机值
	//2.负样本的数目要维持在一个常量 _pm._nn, 而正样本的话随着不断的去掉低于分类阈值的样本而会减少，要选用合理的容器来存放这些信息

	std::list<Dt> p_Dt, n_Dt;
		//组合正样本
	for (int i = 0; i < _pm._np; i++)
	{
		Dt temp;
		temp._weight = 1.0;
		temp._score = 0.0;
		temp._bbox = md->_bbox_origial[i];
		temp._gtshape = md->_gtShape[i];
		temp._lable = 1;
		temp._path = md->_imagsPath["POSITIVE"][i];
		temp._prdshape = ProjectShape(md->_Meanshape.col(0), md->_Meanshape.col(1), temp._bbox); //Initializing the predicted shape with mean shape, it will updated after
		p_Dt.push_back(temp);

	}
		//组合负样本
	for (int i = 0; i < _pm._nn; i++)
	{
		Dt temp = GeNegSamp(md);
		n_Dt.push_back(temp);
	}

	
	//Train
	for (uint16 t = 0; t < _pm._T; t++)//Foreach Stage（Cascade）
	{
		for (uint32 k = 0; k < _pm._K; k++)//Foreach weak classifier (Binary Tree)
		{
			//Computing samples' weight(Not for the first stage) 
			if (0 != t)
			{

			}
		}
	}
}



Dt JDA::GeNegSamp(const _MyData* const md)
{
	time_t current_time;
	current_time = time(0);
	cv::RNG rd(current_time);
	int rdn = rd.uniform(0, _pm._nn - 1);//Selecting a sample from negative set randomly

	/* code */
	Dt result;

	result._bbox.x = rd.uniform(0.2, 0.8); //This scope can be varied  
	result._bbox.y = rd.uniform(0.2, 0.8);
	result._bbox.width = rd.uniform(0.2, 0.5);
	result._bbox.height = rd.uniform(0.2, 0.5);
	result._bbox.ctx = result._bbox.x + result._bbox.width / 2;
	result._bbox.cty = result._bbox.y + result._bbox.height / 2;

	result._weight = 1.0;
	result._score = 0.0;
	result._lable = -1;
	result._path = md->_imagsPath["NEGATIVE"][rdn];
	result._prdshape = ProjectShape(md->_Meanshape.col(0), md->_Meanshape.col(1), result._bbox); 


	return result;
}
