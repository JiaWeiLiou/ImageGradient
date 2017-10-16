#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

/*�Ыئ���*/
void makecolorwheel(vector<Scalar> &colorwheel);

/*�N�Ϥ���H������V�����*/
void DrawMunsellColorSystem(InputArray _field, OutputArray _colorField);

/*�N�Ϥ���H����Ȧǫ����*/
void DrawAbsGraySystem(InputArray _field, OutputArray _grayField);

/*�p������Ϋ�����V���*/
void Differential(InputArray _src, OutputArray _grad_x, OutputArray _grad_y);

/*���X�����Ϋ�����V��׬���׳�*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*�D���j�ȧ��*/
void NonMaximumSuppression(InputArray _gradientField, OutputArray _NMSgradientField);