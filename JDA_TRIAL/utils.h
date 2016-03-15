#ifndef UTILS_____
#define UTILS_____

#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <io.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <map>
#include <time.h>



typedef unsigned int uint32;
typedef unsigned short uint16;
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


const float DMAX_PDM = 3.0;
const float SHAPE_SCALE = 1;

struct PARAMETERS
{
	//Parameters listed in paper
	uint16 _L;
	uint16 _T;
	uint16 _N;
	uint16 _K;

	//
	uint32 _n_n;  //# of negative sample
	uint32 _n_p;  //# of positive sample


	//JDA
	uint16 _n_childTress;
	uint16 _n_deepth;
	uint16 _n_splitFeatures;
	std::vector<float> _radius;


};

struct BBOX
{
	float x, y, width, height, ctx, cty;
	BBOX()
	{
		x = 0; y = 0; width = 0; height = 0; ctx = 0; cty = 0;
	}
};

struct IMGPROPERTY
{
	float width;
	float height;
	IMGPROPERTY(float w, float h) :width(w), height(h){}
};

struct DT
{
	cv::Mat _img; //original image(gray)
	cv::Mat_<float> _gtshape; //ground-truth shape, and equal to meanshape if it is negtive sample 
	cv::Mat_<float> _prdshape; //predicted shape(current shape)��intialized with mean shape
	cv::Mat_<double>_rotation; //The invese transformation matrix
	cv::Mat_<double>_pixDiffFeat; //Only contain current weak classifier's feature�� 
	cv::Mat_<double>_regressionTarget; 
	cv::Mat_<float>_LBF; 


	std::string _className; //��POSITIVE��or��NEGATIVE";
	std::string _path; //the path of origial image

	uint16 _lable; //1 for pos��-1 for neg
	int _index; //laled the origial image's index in MYDATA set
	float _score; //classfy score
	float _weight; //weight
	double _scale; // Reciprocal of scale factor of gtshape alinged to meanshape
	
	BBOX _bbox; //face region 
	

	DT()
	{
		_index = -1;
		_lable = -99;
		_score = 0.0;
		_weight = 0.0;
		_scale = 1;
	}
};

class FeatureLocations
{
public:
	cv::Point2d start;
	cv::Point2d end;
	FeatureLocations(cv::Point2d a, cv::Point2d b){
		start = a;
		end = b;
	}
	FeatureLocations(){
		start = cv::Point2d(0.0, 0.0);
		end = cv::Point2d(0.0, 0.0);
	};
};

enum CorR
{
	CLASSIFICATION = 0, REGRESSION
};

struct ASTANDAR
{
	float _TPR, _FPR, _Theta;
	ASTANDAR(){ _TPR = 0; _FPR = 0; _Theta = 0; }
	bool operator <(const ASTANDAR as) const
	{
		return (as._TPR <= _TPR) && (_FPR <= as._FPR);
	}
};

class Node {
public:
	int _leaf_identity; // used only when it is leaf node, and is unique among the tree
	Node* _left_child;
	Node* _right_child;
	int _samples;
	bool _is_leaf;
	int _depth; // recording current depth
	double _threshold;
	bool _thre_changed;
	CorR _cor;
	double _cscore;
	FeatureLocations _feature_locations;
	int ft_index;
	Node(Node* left, Node* right, double thres, bool leaf);
	Node(Node* left, Node* right, double thres);
	Node();

	ASTANDAR as;
};

class MYDATA
{
private:
	cv::Mat_<float> _ReadPts(const std::string& ptsName, const std::string& type, const int npt);
	
public:
	std::map<std::string, std::vector<std::string>> _imagsPath;
	std::map<std::string, std::vector<IMGPROPERTY>> _imagsProperty;
	std::vector<cv::Mat_<float>>_gtShape;
	std::map<std::string, std::vector<uint32>>_labels;
	std::vector<std::string>_dname; //file name
	std::vector<BBOX> _bbox_origial;
	cv::Mat_<float> _Meanshape;

	void _CalcMeanshape();
	void _DataLoading(const std::string& path, const std::string& type, MYDATA* md, const int n_pt);
	void _GetBbox(const std::vector<cv::Mat_<float>>& shape, const cv::Scalar_<float>& factor, std::vector<BBOX>& bbox_origial);

	~MYDATA();
};

template<typename T> inline
void ReleaseVec(std::vector<T>&input)
{
	std::vector<T>().swap(input);
}

template<typename T> inline
void ScaleVec(std::vector<T>&input)
{
	std::vector<T>(input).swap(input);
}

template<typename T>
T RandNumberUniform(const T low, const T high)
{
	/*time_t current_time;
	current_time = time(0);*/
	cv::waitKey(100);
	cv::RNG rd(cvGetTickCount());
	T rdn = rd.uniform(low, high);
	return rdn;
}

inline void AsignWeight(DT* dt){ dt->_weight = exp(-1 * dt->_lable* dt->_score); };

void drawFeatureP(cv::Mat& image, const cv::Rect& faceRegion, const cv::Mat_<float>&gtp, cv::Scalar sc);
void maxminVec(const cv::Mat_<float>& shape, BBOX& wh);
int randScalar(const int max, int* input);
cv::Mat_<float> ProjectShape(const cv::Mat_<float>& x, const cv::Mat_<float>& y, const BBOX& bbox);
cv::Mat_<float> ReProjection(const cv::Mat_<float>& meanShape, const BBOX& bbox, const cv::Scalar_<float> factor);
void DrawPredictImage(cv::Mat& image, cv::Mat_<float>& shape);
cv::Mat_<float> calcRME(const std::vector<cv::Mat_<float>>&X_updated, const cv::Mat_<float>&gtp_x, const cv::Mat_<float>&gtp_y, int * left_eye, int* right_eye, const int numRbbox, const int numpt);
cv::Mat_<float> calcRME(const std::vector<cv::Mat_<float>>&X_updated, const cv::Mat_<float>&gtp_x, const cv::Mat_<float>&gtp_y, int * left_eye, int* right_eye, const int numRbbox, const int numpt, const cv::Mat_<float>mask);
void getSimilarityTransform(const cv::Mat_<double>& shape_to, const cv::Mat_<double>& shape_from,
	cv::Mat_<double>& rotation, double& scale);
DT* GeNegSamp(MYDATA* const md, const PARAMETERS& pm);

void calcRot_target(const cv::Mat_<float>& ms, DT* dt);




#endif

