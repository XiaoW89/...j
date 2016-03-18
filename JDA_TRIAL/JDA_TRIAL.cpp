// JDA_TRIAL.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "utils.h"
#include "JDA.h"

int _tmain(int argc, _TCHAR* argv[])
{
	//¡­¡­Preparing data¡­¡­//
	std::string dr = "F:\\test";
	const int n_pt = 17;
	MYDATA* md = new MYDATA;
	{
		//Loading data
		md->_DataLoading(dr, "jpg", md, n_pt);

		//Acquiring BBox
		md->_GetBbox(md->_gtShape, cv::Scalar(0, 0, 0, 0), md->_bbox_origial);

		//Calculating Meanshape
		md->_CalcMeanshape();
	}
	
	//¡­¡­Parameters Setting¡­¡­//
	PARAMETERS pm;
	{
		pm._L = n_pt;
		pm._T = 5;
		pm._N = 1700;
		pm._K = pm._N / pm._T;

		pm._n_p = md->_labels["POSITIVE"].size();
		pm._n_n = pm._n_p * 2;

		assert(pm._K % pm._L == 0);
		pm._n_childTress = pm._K / pm._L;
		pm._n_deepth = 3;
		pm._n_splitFeatures = 400;
		pm._radius.push_back(0.4);
		pm._radius.push_back(0.3);
		pm._radius.push_back(0.2);
		pm._radius.push_back(0.1);
		pm._radius.push_back(0.08);

	}

	//¡­¡­Train JDA¡­¡­//
	if (0)
	{
		for (int i = 0; i < md->_gtShape.size(); i++)
		{
			cv::Mat img = cv::imread(md->_imagsPath["POSITIVE"][i]);
			DrawPredictImage(img, md->_gtShape[i]);
			cv::imshow("gt", img);
			cv::waitKey(0);
		}
	}
	
	JDA* jda = new JDA(pm);
	jda->trainJDA(md);

	std::cout << std::endl;
	return 0;


}

