// JDA_TRIAL.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "utils.h"

int _tmain(int argc, _TCHAR* argv[])
{
	//������ȡ������Ϣ����//
	std::string dr = "F:\\test";
	_MyData* md = new _MyData;
	{
		//��ȡ��״
		md->_DataLoading(dr, "jpg", md);

		//�õ�bbox
		md->_GetBbox(md->_gtShape, cv::Scalar(0, 0, 0, 0), md->_bbox_origial);

		//����meanshape
		md->_CalcMeanshape();
	}
	
	//�����������á���//
	_Parameters pm;
	{
		pm._L = 51;
		pm._T = 5;
		pm._N = 2000;
		pm._K = pm._N / pm._T;
	}

	//����ѵ������//

	std::cout << std::endl;
	return 0;


}

