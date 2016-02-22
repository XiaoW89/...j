#include "stdafx.h"
#include "JDA.h"


JDA::JDA(){}
JDA::JDA(const _Parameters& pm){_pm = pm;}
JDA::~JDA(){}

void JDA::train(const _MyData* md)
{
	//We need checkout positive samples and generate negative samples
	//1.���ڸ������ĳ�ȡ������Ϊ�ڳ���ͼƬ��������ɿ�Ȼ���meanshapeͶӰ��ȥ����Ĵ�С��λ�ò���Ϊ����ͼƬ�����ĳ�����������ֵ
	//2.����������ĿҪά����һ������ _pm._nn, ���������Ļ����Ų��ϵ�ȥ�����ڷ�����ֵ������������٣�Ҫѡ�ú���������������Щ��Ϣ

	std::list<Dt> p_Dt, n_Dt;
		//���������
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
		//��ϸ�����
	for (int i = 0; i < _pm._nn; i++)
	{
		Dt temp = GeNegSamp(md);
		n_Dt.push_back(temp);
	}

	
	//Train
	for (uint16 t = 0; t < _pm._T; t++)//Foreach Stage��Cascade��
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
