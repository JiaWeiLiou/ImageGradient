#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

/*創建色環*/
void makecolorwheel(vector<Scalar> &colorwheel);

/*將圖片轉以色環方向場顯示*/
void DrawMunsellColorSystem(InputArray _field, OutputArray _colorField);

/*將圖片轉以絕對值灰度顯示*/
void DrawAbsGraySystem(InputArray _field, OutputArray _grayField);

/*計算水平及垂直方向梯度*/
void Differential(InputArray _src, OutputArray _grad_x, OutputArray _grad_y);

/*結合水平及垂直方向梯度為梯度場*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*非極大值抑制*/
void NonMaximumSuppression(InputArray _gradientField, OutputArray _NMSgradientField);