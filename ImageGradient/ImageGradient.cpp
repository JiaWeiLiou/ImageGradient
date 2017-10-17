// ImageGradient.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include "ImageProcessing.h"
#include <iostream>
#include <opencv2/opencv.hpp>  
#include <cmath>

#define UNKNOWN_FLOW_THRESH 1e9
#define PI 3.14159265359

using namespace std;
using namespace cv;

int main()
{
	std::cout << "Please enter image path : ";
	string infile;
	cin >> infile;

	/*設定輸出文件名*/

	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));							//檔案路徑
	string infilename(infile.substr(pos1 + 1, pos2 - pos1 - 1));		//檔案名稱

	/*載入原圖*/

	Mat srcImage = imread(infile, 0);
	if (!srcImage.data) { printf("Oh，no，讀取srcImage錯誤~！ \n"); return false; }

	/*計算梯度場*/

	Mat grad_x;		//水平方向梯度
	Mat grad_y;		//垂直方向梯度
	//Differential(srcImage, grad_x, grad_y);	//計算梯度場
	Sobel(srcImage, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(srcImage, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);

	Mat gradientField;	//結合水平及垂直方向梯度為梯度場
	GradientField(grad_x, grad_y, gradientField);

	Mat gradient_mag;		//計算梯度幅值(>0)
	Mat gradient_dir;		//計算梯度方向(0-360)
	CalculateGradient(gradientField, gradient_mag, gradient_dir);

	Mat gradientImage_col;		//輸出用色環梯度場
	DrawMunsellColorSystem(gradientField, gradientImage_col);
	string gradOutfile_col = filepath + "\\" + infilename + "_GRAD(color).png";		//梯度場(色環)
	imwrite(gradOutfile_col, gradientImage_col);

	Mat gradientImage_abs;		//輸出用絕對值梯度場
	DrawAbsGraySystem(gradientField, gradientImage_abs);
	string gradOutfile_abs = filepath + "\\" + infilename + "_GRAD(abs).png";		//梯度場(絕對值)
	imwrite(gradOutfile_abs, gradientImage_abs);

	/*非極大值抑制*/

	Mat NMSgradientField;
	NonMaximumSuppression(gradientField, NMSgradientField);

	Mat NMSgradientField_col;	//輸出用非最大值抑制色環梯度場
	DrawMunsellColorSystem(NMSgradientField, NMSgradientField_col);
	string nmsOutfile_col = filepath + "\\" + infilename + "_NMS(color).png";		//非極大值抑制(色環)
	imwrite(nmsOutfile_col, NMSgradientField_col);

	Mat NMSgradientField_abs;	//輸出用非最大值抑制絕對值梯度場
	DrawAbsGraySystem(NMSgradientField, NMSgradientField_abs);
	string nmsOutfile_abs = filepath + "\\" + infilename + "_NMS(abs).png";			//非極大值抑制(絕對值)
	imwrite(nmsOutfile_abs, NMSgradientField_abs);

	//test
	Mat UT;		//上閥值二值化
	threshold(NMSgradientField_abs, UT, 30, 255, THRESH_BINARY);
	Mat label;
	int num = bwlabel(UT, label);

	/*滯後閥值*/
	Mat HTedge;
	HysteresisThreshold(NMSgradientField_abs, HTedge, 80, 40);

	string atOutfile = filepath + "\\" + infilename + "_HT.png";		//非極大值抑制(色環)
	imwrite(atOutfile, HTedge);

    return 0;
}
