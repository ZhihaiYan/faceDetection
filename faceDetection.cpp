#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void faceDetection(Mat image, vector<Rect> &faces)
{
	CascadeClassifier face_cascade;
	face_cascade.load("haarcascade_frontalface_alt2.xml");
	Mat grayImage;
	cvtColor(image, grayImage, CV_BGR2GRAY);
	equalizeHist(grayImage, grayImage);
	face_cascade.detectMultiScale(grayImage, faces, 1.1, 4, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(10, 10),Size(100, 100));

}

//void main()
//{
//	CascadeClassifier face_cascade;
//	face_cascade.load("haarcascade_frontalface_alt2.xml");
//	vector<Rect> faces;
//	Mat faceImage[8];
//	Mat faceResize[8];
//	Mat faceGray;
//	Size imagesize = Size(50, 50);
//	char faceName[20];
//	Mat grayImage;
//	char imagename[50];
//	char imageout[50];
//	int writeNum = 0;
//	int writeNumber = 0;
//	for (int frame = 0; frame < 100; frame++)
//	{
//		sprintf(imagename, "smile/negtive/%04d.pgm", frame+1);
//		Mat image = imread(imagename);
//		Mat image1 = image.clone();
//		cvtColor(image, grayImage, CV_BGR2GRAY);
//		equalizeHist(grayImage, grayImage);
//		face_cascade.detectMultiScale(grayImage, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(100, 100));
//		for (int i = 0; i < faces.size(); i++)
//		{
//			Rect rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
//			faceImage[i] = image(rect);
//			resize(faceImage[i], faceResize[i], imagesize);
//			sprintf(imageout, "facesmile/negtive/%04d.pgm", writeNumber);
//			/*if (writeNum == 22 || writeNum == 51 || writeNum ==6)
//			{
//
//			}
//			else*/
//			if (writeNum == 7 || writeNum == 10 || writeNum ==13 || writeNum == 24 || writeNum == 27 || writeNum == 30 || writeNum == 54)
//			{
//
//			}
//			else
//			{
//				imwrite(imageout, faceResize[i]);
//				writeNumber++;
//			}
//			writeNum++;
//		}
//
//	}
//	
//	//for (int i = 0; i < faces.size(); i++)
//	//{
//	//	Rect rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
//	//	faceImage[i] = image(rect);
//	//	cvtColor(faceImage[i], faceGray[i], CV_BGR2GRAY);
//	//	resize(faceGray[i], faceResize[i], imagesize);
//
//
//	//	Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
//	//	Point pt2(faces[i].x, faces[i].y);
//	//	rectangle(image1, pt1, pt2, cvScalar(0, 255, 0, 0), 3, 8, 0);
//	//	sprintf(faceName,"face%d", i);
//	//	imshow(faceName, faceResize[i]);
//	//}
//	////print the output
//	//imshow("Face Dection, Results", image1);
//	////pause for 33ms
//	//waitKey();
//}