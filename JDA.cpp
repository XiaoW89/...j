#include "stdafx.h"
#include "JDA.h"


JDA::JDA(){}
JDA::JDA(const _Parameters& pm){_pm = pm;}
JDA::~JDA(){}

void JDA::train(const _MyData* md)
{
	//We need checkout positive samples and generate negative samples
	//���ڸ������ĳ�ȡ������Ϊ�ڳ���ͼƬ��������ɿ�Ȼ���meanshapeͶӰ��ȥ����Ĵ�С��λ�ò���Ϊ����ͼƬ�����ĳ�����������ֵ


	//Train
	for (uint16 t = 0; t < _pm._T; t++)//Foreach Stage��Cascade��
	{
		for (uint32 k = 0; k < _pm._K; k++)//Foreach weak classifier (Binary Tree)
		{

		}
	}

}
