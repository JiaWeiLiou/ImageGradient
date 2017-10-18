#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

/*�Ыئ���*/
void makecolorwheel(vector<Scalar> &colorwheel);

/*�N�Ϥ���H������V�����*/
void DrawMunsellColorSystem(InputArray _field, OutputArray _colorField);

/*�N�Ϥ���H����Ȧǫ����*/
void DrawAbsGraySystem(InputArray _field, OutputArray _grayField);

/*�H��t�������G��ܦ����Φǫ׭�*/
void DrawEdgeSystem(InputArray _edge, InputArray _field, OutputArray _edgeField);

/*�p������Ϋ�����V���*/
void Differential(InputArray _src, OutputArray _grad_x, OutputArray _grad_y);

/*���X�����Ϋ�����V��׬���׳�*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*�p���״T�ȤΤ�V*/
void CalculateGradient(InputArray _gradientField, OutputArray _gradient_mag, OutputArray _gradient_dir);

/*�D���j�ȧ��*/
void NonMaximumSuppression(InputArray _gradientField, OutputArray _NMSgradientField);

/*�M��ڵ��I*/
int findroot(int labeltable[], int label);

/*�M��s�q�u*/
int bwlabel(InputArray _binaryImg, OutputArray _labels);

/*����֭�*/
void HysteresisThreshold(InputArray _NMSgradientField_abs, OutputArray _HTedge, int upperThreshold, int lowerThreshold);