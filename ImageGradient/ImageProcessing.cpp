#include "stdafx.h"
#include "ImageProcessing.h"

void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;	//紅色(Red)     至黃色(Yellow)
	int YG = 15;	//黃色(Yellow)  至綠色(Green)
	int GC = 15;	//綠色(Green)   至青色(Cyan)
	int CB = 15;	//青澀(Cyan)    至藍色(Blue)
	int BM = 15;	//藍色(Blue)    至洋紅(Magenta)
	int MR = 15;	//洋紅(Magenta) 至紅色(Red)

	for (int i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (int i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (int i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (int i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (int i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (int i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void DrawMunsellColorSystem(InputArray _field, OutputArray _colorField)
{
	Mat field;
	Mat temp = _field.getMat();
	if (temp.depth() == CV_16S) {
		temp.convertTo(field, CV_32FC2);
	}
	else {
		field = _field.getMat();
	}

	_colorField.create(field.size(), CV_8UC3);
	Mat colorField = _colorField.getMat();

	static vector<Scalar> colorwheel; //Scalar r,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:  
	float maxrad = -1;

	// Find max flow to normalize fx and fy  
	for (int i = 0; i < field.rows; ++i)
	{
		for (int j = 0; j < field.cols; ++j)
		{
			Vec2f field_at_point = field.at<Vec2f>(i, j);
			float fx = field_at_point[0];
			float fy = field_at_point[1];
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}

	for (int i = 0; i < field.rows; ++i)
	{
		for (int j = 0; j < field.cols; ++j)
		{
			uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;
			Vec2f field_at_point = field.at<Vec2f>(i, j);

			float fx = field_at_point[0] / maxrad;
			float fy = field_at_point[1] / maxrad;

			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;    //單位為-1至+1
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;

			float col0 = 0.0f;
			float col1 = 0.0f;
			float col = 0.0f;
			for (int b = 0; b < 3; b++)
			{
				col0 = colorwheel[k0][b] / 255.0f;
				col1 = colorwheel[k1][b] / 255.0f;
				col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius  
				else
					col = col;  //out of range
				data[2 - b] = (int)(255.0f * col);
			}
		}
	}
}

void DrawAbsGraySystem(InputArray _field, OutputArray _grayField)
{
	Mat field;
	Mat temp = _field.getMat();
	if (temp.depth() == CV_16S) {
		temp.convertTo(field, CV_32FC2);
	}
	else {
		field = _field.getMat();
	}

	_grayField.create(field.size(), CV_8UC1);
	Mat grayField = _grayField.getMat();

	// determine motion range:  
	float maxvalue = -1;

	// Find max flow to normalize fx and fy  
	for (int i = 0; i < field.rows; ++i)
		for (int j = 0; j < field.cols; ++j) {
			Vec2f field_at_point = field.at<Vec2f>(i, j);
			float fx = field_at_point[0];
			float fy = field_at_point[1];
			float absvalue = sqrt(fx * fx + fy * fy);
			maxvalue = maxvalue > absvalue ? maxvalue : absvalue;
		}

	for (int i = 0; i < field.rows; ++i)
		for (int j = 0; j < field.cols; ++j) {
			Vec2f field_at_point = field.at<Vec2f>(i, j);
			float fx = field_at_point[0];
			float fy = field_at_point[1];
			float absvalue = sqrt(fx * fx + fy * fy);
			grayField.at<uchar>(i, j) = (char) (absvalue*255.0f / maxvalue);
		}
}

void Differential(InputArray _src, OutputArray _grad_x, OutputArray _grad_y) {

	Mat src = _src.getMat();
	CV_Assert(src.depth() == CV_8U);

	_grad_x.create(src.size(), CV_16S);
	Mat grad_x = _grad_x.getMat();

	_grad_y.create(src.size(), CV_16S);
	Mat grad_y = _grad_y.getMat();

	Mat srcRef;
	copyMakeBorder(src, srcRef, 0, 1, 1, 0, BORDER_REPLICATE);
	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			grad_x.at<short>(i, j) = -(short)srcRef.at<uchar>(i, j) + (short)srcRef.at<uchar>(i, j + 1);
			grad_y.at<short>(i, j) = -(short)srcRef.at<uchar>(i, j) + (short)srcRef.at<uchar>(i + 1, j);
		}
}

void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField) {

	Mat grad_x = _grad_x.getMat();
	CV_Assert(grad_x.depth() == CV_16S);

	Mat grad_y = _grad_y.getMat();
	CV_Assert(grad_y.depth() == CV_16S);

	_gradientField.create(grad_y.rows, grad_x.cols, CV_16SC2);
	Mat gradientField = _gradientField.getMat();

	for (int i = 0; i < grad_y.rows; ++i)
		for (int j = 0; j < grad_x.cols; ++j)
		{
			gradientField.at<Vec2s>(i, j)[0] = -grad_x.at<short>(i, j);
			gradientField.at<Vec2s>(i, j)[1] = -grad_y.at<short>(i, j);
		}

}

void NonMaximumSuppression(InputArray _gradientField, OutputArray _NMSgradientField)
{
	Mat gradientField = _gradientField.getMat();
	CV_Assert(gradientField.depth() == CV_16S);

	_NMSgradientField.create(gradientField.size(), CV_16SC2);
	Mat NMSgradientField = _NMSgradientField.getMat();

	Mat gradientFieldRef;
	copyMakeBorder(gradientField, gradientFieldRef, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0, 0));

	float theta = 0.0f;			//目前像素的方向
	int amplitude = 0;			//目前像素的幅值
	int amplitude1 = 0;			//鄰域像素1的幅值
	int amplitude2 = 0;			//鄰域像素2的幅值
	float A1 = 0.0f;			//上臨域1幅值
	float A2 = 0.0f;			//上臨域2幅值
	float B1 = 0.0f;			//下臨域1幅值
	float B2 = 0.0f;			//下臨域2幅值
	float alpha = 0.0f;			//比例係數

	for (int i = 0; i < gradientField.rows; ++i)
		for (int j = 0; j < gradientField.cols; ++j)
		{
			theta = fastAtan2(gradientField.at<Vec2s>(i, j)[1], gradientField.at<Vec2s>(i, j)[0]);
			amplitude = pow(gradientField.at<Vec2s>(i, j)[0], 2) + pow(gradientField.at<Vec2s>(i, j)[1], 2);
			if ((theta >= 0.0f && theta < 45.0f) || (theta >= 180.0f && theta < 225.0f))
			{
				alpha = tan(theta* CV_PI / 180.0);
				A1 = pow(gradientFieldRef.at<Vec2s>(i + 1, j + 2)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i + 1, j + 2)[1], 2);
				A2 = pow(gradientFieldRef.at<Vec2s>(i + 2, j + 2)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i + 2, j + 2)[1], 2);
				B1 = pow(gradientFieldRef.at<Vec2s>(i + 1, j)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i + 1, j)[1], 2);
				B2 = pow(gradientFieldRef.at<Vec2s>(i, j)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i, j)[1], 2);

			}
			else if ((theta >= 45.0f && theta < 90.0f) || (theta >= 225.0f && theta < 270.0f))
			{
				alpha = tan((90.0f - theta)* CV_PI / 180.0);
				A1 = pow(gradientFieldRef.at<Vec2s>(i + 1, j + 1)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i + 1, j + 1)[1], 2);
				A2 = pow(gradientFieldRef.at<Vec2s>(i + 2, j + 2)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i + 2, j + 2)[1], 2);
				B1 = pow(gradientFieldRef.at<Vec2s>(i, j + 1)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i, j + 1)[1], 2);
				B2 = pow(gradientFieldRef.at<Vec2s>(i, j)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i, j)[1], 2);
			}
			else if ((theta >= 90.0f && theta < 135.0f) || (theta >= 270.0f && theta < 315.0f))
			{
				alpha = tan((theta - 90.0f)* CV_PI / 180.0);
				A1 = pow(gradientFieldRef.at<Vec2s>(i + 1, j + 1)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i + 1, j + 1)[1], 2);
				A2 = pow(gradientFieldRef.at<Vec2s>(i + 2, j)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i + 2, j)[1], 2);
				B1 = pow(gradientFieldRef.at<Vec2s>(i, j + 1)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i, j + 1)[1], 2);
				B2 = pow(gradientFieldRef.at<Vec2s>(i, j + 2)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i, j + 2)[1], 2);
			}
			else if ((theta >= 135.0f && theta < 180.0f) || (theta >= 315.0f && theta < 360.0f))
			{
				alpha = tan((180.0f - theta)* CV_PI / 180.0);
				A1 = pow(gradientFieldRef.at<Vec2s>(i + 1, j)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i + 1, j)[1], 2);
				A2 = pow(gradientFieldRef.at<Vec2s>(i + 2, j)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i + 2, j)[1], 2);
				B1 = pow(gradientFieldRef.at<Vec2s>(i + 1, j + 2)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i + 1, j + 2)[1], 2);
				B2 = pow(gradientFieldRef.at<Vec2s>(i, j + 2)[0], 2) + pow(gradientFieldRef.at<Vec2s>(i, j + 2)[1], 2);
			}

			amplitude1 = A1*(1 - alpha) + A2*alpha;
			amplitude2 = B1*(1 - alpha) + B2*alpha;

			if (amplitude >= amplitude1 && amplitude >= amplitude2) {
				NMSgradientField.at<Vec2s>(i, j)[0] = gradientField.at<Vec2s>(i, j)[0];
				NMSgradientField.at<Vec2s>(i, j)[1] = gradientField.at<Vec2s>(i, j)[1];
			}
			else {
				NMSgradientField.at<Vec2s>(i, j)[0] = 0.0f;
				NMSgradientField.at<Vec2s>(i, j)[1] = 0.0f;
			}

		}
}