#include "stdafx.h"
#include "JDA.h"



JDA::JDA(){}
JDA::JDA(const PARAMETERS& pm){_pm = pm;}
JDA::~JDA(){}

void JDA::trainJDA(MYDATA* const md)
{
	//********SETP1. We need checkout positive samples and generate negative samples********
	//1.���ڸ������ĳ�ȡ������Ϊ�ڳ���ͼƬ��������ɿ�Ȼ���meanshapeͶӰ��ȥ����Ĵ�С��λ�ò���Ϊ����ͼƬ�����ĳ�����������ֵ
	//2.����������ĿҪά����һ������ _pm._nn, ���������Ļ����Ų��ϵ�ȥ�����ڷ�����ֵ������������٣�Ҫѡ�ú���������������Щ��Ϣ

	std::deque<DT*> p_Dt, n_Dt;
		//padding positive data

	for (int i = 0; i < _pm._n_p; i++)
	{
		float w = md->_imagsProperty["POSITIVE"][i].width;
		float h = md->_imagsProperty["POSITIVE"][i].height;

		DT* temp = new DT;
		temp->_bbox = md->_bbox_origial[i];
		temp->_path = md->_imagsPath["POSITIVE"][i];
		
		temp->_gtshape = md->_gtShape[i];
		temp->_prdshape = ReProjection(md->_Meanshape, temp->_bbox, cv::Scalar_<float>(1, 1, 1, 1)); //Initializing the predicted shape with mean shape, it will updated after
		temp->_pixDiffFeat.create(1, _pm._n_splitFeatures);
		temp->_className = "POSITIVE";
		

		temp->_lable_true = 1;
		temp->_label_preditced = 0;
		temp->_index = i;
		temp->_cscore = 0.0;
		temp->_weight = 1.0;
		temp->_scale = 1;

		p_Dt.push_back(temp);

	}
		//padding negative data

	//int cc = 1;

	for (int i = 0; i < _pm._n_n; i++)
	{
		DT* temp = GeNegSamp(md, _pm);
		n_Dt.push_back(temp);
		//cc++;
		//std::cout << cc << std::endl;
	}

	//********STEP2. Train JDA********
	std::vector<RandomForest>cascade;
	_shape_param_set.clear();

	for (uint16 i = 0; i < _pm._T; i++)//Foreach Stage��Cascade��
	{
		//********STEP 2.1 : Compute similarity transform matrix , scale factor and regression target********
		std::cout << "calculate regression targets" << std::endl;

		for (int j = 0; j < p_Dt.size(); j++)
			calcRot_target(md->_Meanshape, p_Dt[j]);

		for (int j = 0; j < n_Dt.size(); j++)
			calcRot_target(md->_Meanshape, n_Dt[j]);

		//********STEP 2.2 : Learn each stage********
		RandomForest rf(_pm, i);
		rf.TrainForest(md, _pm, _shape_param_set, p_Dt, n_Dt, cascade);
		cascade.push_back(rf);


		//********STEP 2.3 : Learn the shape increments of all leaves (global increments learning)********
			//Extract global LBP for each positive sample
		for (int j = 0; j < p_Dt.size(); j++)
		{
			calcRot_target(md->_Meanshape, p_Dt[j]);
			GetGlobalLBF(md, rf, p_Dt[j]);
		}
			//LibLinear
		parameter param;
		param.solver_type = L2R_L2LOSS_SVR_DUAL;
		//param.C = 0.0005;
		param.C = 10 / (float)p_Dt.size();
		param.eps = 0.001;
		param.p = 0;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;

		problem pb;
		pb.n = p_Dt[0]->_LBF.cols;
		pb.l = p_Dt.size();
		pb.bias = 1;
		pb.x = Malloc(feature_node*, pb.l);
		pb.y = Malloc(double, pb.l);

		for (int j = 0; j < pb.l; j++)
		{
			pb.x[j] = Malloc(feature_node, pb.n + 1);
#pragma omp parallel for
			for (int z = 0; z < pb.n + 1; z++)
			{
				if (z == (pb.n))
					(pb.x[j])[z].index = -1;
				else
				{
					(pb.x[j])[z].index = z + 1;
					(pb.x[j])[z].value = p_Dt[j]->_LBF(0, z);
				}
			}
		}

		uint16 s = 0;
		model* md;
		cv::Mat_<float>shape_param(this->_pm._L * 2, pb.n);

		std::cout << "ѵ����������" << std::endl;
		while (s < this->_pm._L * 2)
		{
#pragma omp parallel for
			for (int z = 0; z < p_Dt.size(); z++)
			{
				if (s < this->_pm._L)
					pb.y[z] = p_Dt[z]->_regressionTarget.col(0)(s);
				else
					pb.y[z] = p_Dt[z]->_regressionTarget.col(0)(s - this->_pm._L);
			}
			md = train(&pb, &param);
			for (uint16 j = 0; j < pb.n; j++)
				shape_param(s, j) = md->w[j];
			s++;
		}

		//********STEP 2.4 : Update shape for all positive samples (include all false positive samples)********
	
		std::cout << "����������״��" << std::endl;

		for (int j = 0; j < p_Dt.size(); j++)
			UpdateShape(shape_param, p_Dt[j]);
		for (int j = 0; j < n_Dt.size(); j++)
			UpdateShape(shape_param, n_Dt[j]);


		std::cout << "������" << std::endl;
		//calcRME(this->X0, this->gtp_x, this->gtp_y, this->ps.right_eye, this->ps.left_eye, this->ps.numRbbox, this->ps.numPt);
		for (int j = 0; j < pb.l; j++)
			free(pb.x[j]);
		free(pb.x);
		free(pb.y);
		free_and_destroy_model(&md);

		_shape_param_set.push_back(shape_param);
	}
	//
}


