//
//  main.cpp
//  http
//
//  Created by smile on 2018/4/4.
//  Copyright © 2018年 smile. All rights reserved.
//

#include <iostream>
using namespace std;

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "face_alignment.h"
#include "face_detection.h"
#include "face_identification.h"
#include "recognizer.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "math_functions.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace seeta;

// 定义测试数据和模块地址
std::string MODEL_DIR = "./model/";

// 比对两个人脸相似度
float compare(std::string img1, std::string img2);

// 判断是否有人脸存在，返回人脸数
int32_t detect(std::string img_path);

int main(int argc, const char *argv[])
{
	if (argc < 2) {
		cout << "Usage: " << argv[0] << " cmd [option]" << endl;
		return -1;
	}
	// 比较的两个图片地址
	std::string argv1 = argv[1];
	if (argv1 == "compare") {
		if (argc < 4) {
			cout << "Usage: " << argv[0] << " compare img1 img2" << endl;
			return -1;
		}
		std::string img1 = argv[2];
		std::string img2 = argv[3];
		// 调用比较两个图片相似度
		float sim = compare(img1, img2);
		std::cout << sim << endl;
	} else if (argv1 == "detect") {
		if (argc < 3) {
			cout << "Usage: " << argv[0] << " detect img" << endl;
			return -1;
		}
		std::string img_path = argv[2];
		int32_t num_face = detect(img_path);

		cout << num_face << endl;
	} else {
		cout << "Usage: " << argv[0] << " compare | detect" << endl;
		return -1;
	}

	return 0;
}

// 比对两个人脸相似度
float compare(std::string img1, std::string img2)
{
	seeta::FaceDetection detector((MODEL_DIR + "seeta_fd_frontal_v1.0.bin").c_str());

	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model
	seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());
	// Initialize face Identification model
	FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());

	// load image
	cv::Mat gallery_img_color = cv::imread(img1, 1);
	cv::Mat gallery_img_gray;
	cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);

	cv::Mat probe_img_color = cv::imread(img2, 1);
	cv::Mat probe_img_gray;
	cv::cvtColor(probe_img_color, probe_img_gray, CV_BGR2GRAY);

	ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
	gallery_img_data_color.data = gallery_img_color.data;

	ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
	gallery_img_data_gray.data = gallery_img_gray.data;

	ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
	probe_img_data_color.data = probe_img_color.data;

	ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
	probe_img_data_gray.data = probe_img_gray.data;

	// Detect faces
	std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
	int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());

	std::vector<seeta::FaceInfo> probe_faces = detector.Detect(probe_img_data_gray);
	int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());

	if (gallery_face_num == 0 || probe_face_num == 0) {
		// std::cout << "Faces are not detected.";
		return 0;
	}

	// Detect 5 facial landmarks
	seeta::FacialLandmark gallery_points[5];
	point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);

	seeta::FacialLandmark probe_points[5];
	point_detector.PointDetectLandmarks(probe_img_data_gray, probe_faces[0], probe_points);

	// Extract face identity feature
	float gallery_fea[2048];
	float probe_fea[2048];
	face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
	face_recognizer.ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);

	// Caculate similarity of two faces
	float sim = face_recognizer.CalcSimilarity(gallery_fea, probe_fea);
	// std::cout << sim << endl;

	return sim;
}

// 判断是否有人脸存在，返回人脸数
int32_t detect(std::string img_path)
{
	std::string modelPath = MODEL_DIR + "seeta_fd_frontal_v1.0.bin";
	char const *modelPathChar = modelPath.c_str();
	seeta::FaceDetection detector(modelPathChar);

	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	cv::Mat img_gray;

	if (img.channels() != 1)
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = img;

	seeta::ImageData img_data;
	img_data.data = img_gray.data;
	img_data.width = img_gray.cols;
	img_data.height = img_gray.rows;
	img_data.num_channels = 1;

	std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);

	int32_t num_face = static_cast<int32_t>(faces.size());
	return num_face;
}