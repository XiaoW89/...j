// JDA_TRIAL.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "utils.h"

int _tmain(int argc, _TCHAR* argv[])
{
	//……读取数据信息……//
	std::string dr = "F:\\test";
	_MyData* md = new _MyData;
	{
		//读取形状
		md->_DataLoading(dr, "jpg", md);

		//得到bbox
		md->_GetBbox(md->_gtShape, cv::Scalar(0, 0, 0, 0), md->_bbox_origial);

		//计算meanshape
		md->_CalcMeanshape();
	}
	
	//……参数设置……//
	_Parameters pm;
	{
		pm._L = 51;
		pm._T = 5;
		pm._N = 2000;
		pm._K = pm._N / pm._T;
	}

	//……训练……//

	std::cout << std::endl;
	return 0;


}

