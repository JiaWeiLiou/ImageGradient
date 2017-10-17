// ImageGradient.cpp : �w�q�D���x���ε{�����i�J�I�C
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

	/*�]�w��X���W*/

	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));							//�ɮ׸��|
	string infilename(infile.substr(pos1 + 1, pos2 - pos1 - 1));		//�ɮצW��

	/*���J���*/

	Mat srcImage = imread(infile, 0);
	if (!srcImage.data) { printf("Oh�Ano�AŪ��srcImage���~~�I \n"); return false; }

	/*�p���׳�*/

	Mat grad_x;		//������V���
	Mat grad_y;		//������V���
	//Differential(srcImage, grad_x, grad_y);	//�p���׳�
	Sobel(srcImage, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(srcImage, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);

	Mat gradientField;	//���X�����Ϋ�����V��׬���׳�
	GradientField(grad_x, grad_y, gradientField);

	Mat gradient_mag;		//�p���״T��(>0)
	Mat gradient_dir;		//�p���פ�V(0-360)
	CalculateGradient(gradientField, gradient_mag, gradient_dir);

	Mat gradientImage_col;		//��X�Φ�����׳�
	DrawMunsellColorSystem(gradientField, gradientImage_col);
	string gradOutfile_col = filepath + "\\" + infilename + "_GRAD(color).png";		//��׳�(����)
	imwrite(gradOutfile_col, gradientImage_col);

	Mat gradientImage_abs;		//��X�ε���ȱ�׳�
	DrawAbsGraySystem(gradientField, gradientImage_abs);
	string gradOutfile_abs = filepath + "\\" + infilename + "_GRAD(abs).png";		//��׳�(�����)
	imwrite(gradOutfile_abs, gradientImage_abs);

	/*�D���j�ȧ��*/

	Mat NMSgradientField;
	NonMaximumSuppression(gradientField, NMSgradientField);

	Mat NMSgradientField_col;	//��X�ΫD�̤j�ȧ�������׳�
	DrawMunsellColorSystem(NMSgradientField, NMSgradientField_col);
	string nmsOutfile_col = filepath + "\\" + infilename + "_NMS(color).png";		//�D���j�ȧ��(����)
	imwrite(nmsOutfile_col, NMSgradientField_col);

	Mat NMSgradientField_abs;	//��X�ΫD�̤j�ȧ���ȱ�׳�
	DrawAbsGraySystem(NMSgradientField, NMSgradientField_abs);
	string nmsOutfile_abs = filepath + "\\" + infilename + "_NMS(abs).png";			//�D���j�ȧ��(�����)
	imwrite(nmsOutfile_abs, NMSgradientField_abs);

	//test
	Mat UT;		//�W�֭ȤG�Ȥ�
	threshold(NMSgradientField_abs, UT, 30, 255, THRESH_BINARY);
	Mat label;
	int num = bwlabel(UT, label);

	/*����֭�*/
	Mat HTedge;
	HysteresisThreshold(NMSgradientField_abs, HTedge, 80, 40);

	string atOutfile = filepath + "\\" + infilename + "_HT.png";		//�D���j�ȧ��(����)
	imwrite(atOutfile, HTedge);

    return 0;
}
