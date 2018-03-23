//
//  main.cpp
//  threshold
//
//  Created by 杨子荣 on 2017/8/16.
//  Copyright © 2017年 zeroyoung. All rights reserved.
//  在原图上识别绘制瞳孔位置
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
//#define WinSource "source"
#define WinFace "face"
//#define WinFaceGray "faceGray"
#define WinPupilL "PupilL"
#define WinPupilR "PupilR"

//人脸级联分类器地址
string face_cascade_name = "/usr/local/Cellar/opencv/2.4.13.2_1/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
string nested_cascade_name = "/usr/local/Cellar/opencv/2.4.13.2_1/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade,nested_cascade; //人脸级联分类器
double scale=1;
bool tryflip = false;//这个参数的含义？

//瞳孔精定位  返回一个point，相对位置，或者struct，两者的相对，from center 2 pupil
void drawPupil(Mat face,Point center,int radius){
    int g_nGaussianBlurValue = 11;//高斯滤波参数值
    int g_nThresholdValue = 25;//二值化的阈值，灰度图在0-255  暗室情况下，最好的效果是20
    //int g_nThresholdType = 0;//二值化的模式 0-4
    vector<vector<Point> > g_vContours;
    vector<Vec4i> g_vHierarchy;
    Mat grayImg,gaussImg,dstImg;
    vector<Point2f> holeCenter(6);//六个孔的中点
    Moments mu;
    Point2f mc;
    //100是ROI边长，需要进一步优化
    Mat Pupil=face(Rect(center.x-radius,center.y-radius,100,100));
    
    cvtColor(Pupil, grayImg, CV_BGR2GRAY);//转换为灰度图
    GaussianBlur(grayImg, gaussImg, Size(g_nGaussianBlurValue*2+1,g_nGaussianBlurValue*2+1), 0,0);
    threshold(gaussImg, dstImg, g_nThresholdValue, 255,THRESH_BINARY);
    findContours(dstImg,g_vContours,g_vHierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
    vector<vector<Point> >hull(g_vContours.size());
    
    //v2.0
    for(int i = 0; i < g_vContours.size(); i++){
        Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours(Pupil, g_vContours, i, color,1,8,g_vHierarchy);
        mu = moments(g_vContours[i],true);
        mc = Point2f(static_cast<float>(mu.m10/mu.m00),static_cast<float>(mu.m01/mu.m00));
        //cout<<"轮廓面积大小："<<mu.m00<<"  "<<"中心位置"<<mc<<endl;
        
        //根据轮廓大小，识别出代表眼球的轮廓，并找出瞳孔中心
        if(mu.m00>200&&mu.m00<1000)
            circle(Pupil, mc, 3, Scalar(255,255,255));//轮廓中心
    }
}

//人脸检测+人眼检测
void detectAndDisplay( Mat face ){//人脸检测显示，人眼中心
    int i=0;
    double t=0;
    vector<Rect> faces,faces2;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    
    Mat face_gray,smallImg( cvRound (face.rows/scale), cvRound(face.cols/scale), CV_8UC1 ); ;
    cvtColor( face, face_gray, CV_BGR2GRAY );  //rgb类型转换为灰度类型
    resize( face_gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );   //直方图均衡化
    
    t = (double)cvGetTickCount();
    
    //主要函数，
    face_cascade.detectMultiScale( smallImg, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );//更改了Size
    
    if(tryflip){
        flip(smallImg,smallImg,1);
        face_cascade.detectMultiScale( smallImg, faces2, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ ) {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    
    t=(double)cvGetTickCount()-t;
    //检测时间
    //printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ ){
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;
        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 ){
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            circle( face, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( face, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                      cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                      color, 3, 8, 0);
        if( nested_cascade.empty())//人眼检测
            continue;
        smallImgROI = smallImg(*r);
        nested_cascade.detectMultiScale( smallImgROI, nestedObjects,
                                        1.1, 2, 0
                                        //|CV_HAAR_FIND_BIGGEST_OBJECT
                                        //|CV_HAAR_DO_ROUGH_SEARCH
                                        //|CV_HAAR_DO_CANNY_PRUNING
                                        |CV_HAAR_SCALE_IMAGE
                                        ,
                                        Size(30, 30) );
        //眼部轮廓
        for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ ){
            Scalar color = colors[i%8];
            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
            radius = cvRound((nr->width + nr->height)*0.25*scale);
            circle( face, center, radius, color, 3, 8, 0 );//眼部
            drawPupil(face,center,radius);//绘制瞳孔位置
            int radius2=2;
            circle( face, center, radius2, color, 3, 8, 0 );//眼中心
        }
        
    }
    
    //脸部圆
    for( int i = 0; i < faces.size(); i++ ){
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( face, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 0), 2,7, 0 );
    }
    //imshow(WinFaceGray, face_gray );
    imshow(WinFace, face );
}

int main(int argc, const char * argv[]) {
    VideoCapture capture(0);//选择摄像头
    //namedWindow(WinSource,WINDOW_NORMAL);//定义窗口   WINDOW_NORMAL是窗口可调大小，但是是从左上角开始算的
    Mat frame;//定义一帧
    while(1){//进入循环
        capture>>frame;//获取视频中的一帧图像
        frame=frame(Rect(400,150,500,500));
        //imshow(WinSource,frame);
        
        //判断加载分类器是否成功
        if( !face_cascade.load( face_cascade_name ) ){
            printf("级联分类器1错误，可能未找到文件，拷贝该文件到工程目录下！\n");
            return -1;
        }
        if( !nested_cascade.load( nested_cascade_name ) ){
            printf("级联分类器2错误，可能未找到文件，拷贝该文件到工程目录下！\n");
            return -1;
        }
        detectAndDisplay(frame);//检测人脸、瞳孔等
        waitKey(5);//等待 必须要有这一句，不然无法显示图片  定义视频刷新率
    }
    return 0;
}
