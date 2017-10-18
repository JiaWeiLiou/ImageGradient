#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

/*創建色環*/
void makecolorwheel(vector<Scalar> &colorwheel);

/*將圖片轉以色環方向場顯示*/
void DrawMunsellColorSystem(InputArray _field, OutputArray _colorField);

/*將圖片轉以絕對值灰度顯示*/
void DrawAbsGraySystem(InputArray _field, OutputArray _grayField);

/*以邊緣偵測結果顯示色環或灰度值*/
void DrawEdgeSystem(InputArray _edge, InputArray _field, OutputArray _edgeField);

/*計算水平及垂直方向梯度*/
void Differential(InputArray _src, OutputArray _grad_x, OutputArray _grad_y);

/*結合水平及垂直方向梯度為梯度場*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*計算梯度幅值及方向*/
void CalculateGradient(InputArray _gradientField, OutputArray _gradient_mag, OutputArray _gradient_dir);

/*非極大值抑制*/
void NonMaximumSuppression(InputArray _gradientField, OutputArray _NMSgradientField);

/*尋找根結點*/
int findroot(int labeltable[], int label);

/*尋找連通線*/
int bwlabel(InputArray _binaryImg, OutputArray _labels);

/*滯後閥值*/
void HysteresisThreshold(InputArray _NMSgradientField_abs, OutputArray _HTedge, int upperThreshold, int lowerThreshold);