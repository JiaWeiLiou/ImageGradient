// ImageGradient.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"

#include <iostream>
#include <opencv2/opencv.hpp>  
#include <cmath>

#define UNKNOWN_FLOW_THRESH 1e9

using namespace std;
using namespace cv;

void makecolorwheel(vector<Scalar> &colorwheel);
void drawMunsellColorSystem(Mat flow, Mat &color, double maxrad);

int main()
{
	std::cout << "Please enter image path : ";
	string infile;
	cin >> infile;

	/*設定輸出文件名*/
	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));
	string infile_name(infile.substr(pos1 + 1, pos2 - pos1 - 1));
	string gradientoutfile;
	gradientoutfile = filepath + "\\" + infile_name + "_Gradient.JPG";

	// 載入原圖  
	Mat srcImage = imread(infile, 0);
	if (!srcImage.data) { printf("Oh，no，讀取srcImage錯誤~！ \n"); return false; }

	Mat grad_x;
	Mat grad_y;

	Sobel(srcImage, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(srcImage, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);

	/*梯度場*/
	Mat gradient = Mat(srcImage.size(), CV_32FC2);
	
	for (int i = 0; i < srcImage.rows; ++i)
		for (int j = 0; j < srcImage.cols; ++j)
		{
			gradient.at<Vec2f>(i, j)[0] = (double)grad_x.at<short>(i, j);
			gradient.at<Vec2f>(i, j)[1] = (double)grad_y.at<short>(i, j);
		}

	Mat ColorGradientImage;
	int radius = 255;
	drawMunsellColorSystem(gradient, ColorGradientImage, radius);
	imwrite(gradientoutfile, ColorGradientImage);

    return 0;
}

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

void drawMunsellColorSystem(Mat flow, Mat &color, double maxrad)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //Scalar r,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:  
	//float maxrad = -1;

	// Find max flow to normalize fx and fy  
	//for (int i = 0; i < flow.rows; ++i)
	//{
	//	for (int j = 0; j < flow.cols; ++j)
	//	{
	//		Vec2f flow_at_point = flow.at<Vec2f>(i, j);
	//		float fx = flow_at_point[0];
	//		float fy = flow_at_point[1];
	//		if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
	//			continue;
	//		float rad = sqrt(fx * fx + fy * fy);
	//		maxrad = maxrad > rad ? maxrad : rad;
	//	}
	//}

	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;    //單位為-1至+1
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
			//f = 0; // uncomment to see original color wheel  

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius  
				else
					col = col;  //out of range
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}
