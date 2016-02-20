#ifndef UTILS_____
#define UTILS_____

#include <vector>
#include <iostream>
#include <fstream>
#include <io.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <map>

typedef unsigned int uint32;
typedef unsigned short uint16;

enum _ClassID
{
	POSITIVE = 0, NEGATIVE
};

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

const float DMAX_PDM = 3.0;
const float SHAPE_SCALE = 1;


struct _BBOX
{
	float x, y, width, height, ctx, cty;
	_BBOX()
	{
		x = 0; y = 0; width = 0; height = 0; ctx = 0; cty = 0;
	}
};

class _MyData
{
private:
	cv::Mat_<float> _Meanshape;
	cv::Mat_<float> _ReadPts(const std::string& ptsName, const std::string& type, const int npt);
	
public:
	std::map<std::string, std::vector<std::string>> _imagsPath;
	std::vector<cv::Mat_<float>>_gtShape;
	std::map<std::string, std::vector<uint32>>_labels;
	std::vector<std::string>_dname; //ÎÄ¼þ¼ÐÃû
	std::vector<_BBOX> _bbox_origial;

	void _CalcMeanshape();
	void _DataLoading(const std::string& path, const std::string& type, _MyData* md);
	void _GetBbox(const std::vector<cv::Mat_<float>>& shape, const cv::Scalar_<float>& factor, std::vector<_BBOX>& bbox_origial);

	~_MyData();
};

struct _Parameters
{
	uint16 _L;
	uint16 _T;
	uint16 _N;
	uint16 _K;
};




void drawFeatureP(cv::Mat& image, const cv::Rect& faceRegion, const cv::Mat_<float>&gtp, cv::Scalar sc);
void maxminVec(const cv::Mat_<float>& shape, _BBOX& wh);
int randScalar(const int max, int* input);
cv::Mat_<float> ProjectShape(const cv::Mat_<float>& x, const cv::Mat_<float>& y, const _BBOX& bbox);
cv::Mat_<float> ReProjection(const cv::Mat_<float>& meanShape, const _BBOX& bbox, const cv::Scalar_<float> factor);
void DrawPredictImage(cv::Mat& image, cv::Mat_<float>& shape);
cv::Mat_<float> calcRME(const std::vector<cv::Mat_<float>>&X_updated, const cv::Mat_<float>&gtp_x, const cv::Mat_<float>&gtp_y, int * left_eye, int* right_eye, const int numRbbox, const int numpt);
cv::Mat_<float> calcRME(const std::vector<cv::Mat_<float>>&X_updated, const cv::Mat_<float>&gtp_x, const cv::Mat_<float>&gtp_y, int * left_eye, int* right_eye, const int numRbbox, const int numpt, const cv::Mat_<float>mask);


#endif

