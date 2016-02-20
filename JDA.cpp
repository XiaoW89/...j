#include "stdafx.h"
#include "JDA.h"


JDA::JDA(){}
JDA::JDA(const _Parameters& pm){_pm = pm;}
JDA::~JDA(){}

void JDA::train(const _MyData* md)
{
	//We need checkout positive samples and generate negative samples
	//关于负样本的抽取，方法为在场景图片中随机生成框，然后把meanshape投影进去，框的大小和位置参数为场景图片长宽的某个比例的随机值


	//Train
	for (uint16 t = 0; t < _pm._T; t++)//Foreach Stage（Cascade）
	{
		for (uint32 k = 0; k < _pm._K; k++)//Foreach weak classifier (Binary Tree)
		{

		}
	}

}
