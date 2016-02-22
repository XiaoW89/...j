// JDA_TRIAL.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "utils.h"

int _tmain(int argc, _TCHAR* argv[])
{
	//¡­¡­Preparing data¡­¡­//
	std::string dr = "F:\\test";
	_MyData* md = new _MyData;
	{
		//Loading data
		md->_DataLoading(dr, "jpg", md);

		//Acquiring BBox
		md->_GetBbox(md->_gtShape, cv::Scalar(0, 0, 0, 0), md->_bbox_origial);

		//Calculating Meanshape
		md->_CalcMeanshape();


	}
	
	//¡­¡­Parameters Setting¡­¡­//
	_Parameters pm;
	{
		pm._L = 51;
		pm._T = 5;
		pm._N = 2000;
		pm._K = pm._N / pm._T;

		pm._np = md->_labels["POSITIVE"].size();
		pm._nn = pm._np * 10;


	}

	//¡­¡­Train JDA¡­¡­//

	for (int i = 0; i < md->_gtShape.size();i++)
	{
		cv::Mat img = cv::imread(md->_imagsPath["POSITIVE"][i]);
		DrawPredictImage(img, md->_gtShape[i]);
		cv::imshow("gt", img);
		cv::waitKey(0);
	}
	

	std::cout << std::endl;
	return 0;


}

