#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <ctime>

using namespace std;
using namespace cv;

int xx = 0;
int yy = 0;

int xxx = 0;
int yyy = 0;

int counter = 0;
long int timer;


Rect drawing(int x, int y, int x2, int y2)
{
    Rect rect(x, y, x2, y2);
    return rect;
}

long int unix_timestamp()
{
    time_t t = std::time(0);
    long int now = static_cast<long int> (t);
    return now;
}

void CallBackFunck(int event, int x, int y, int flags, void*userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        xxx = xx;
        yyy = yy;

        xx = x;
        yy = y;

        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                   CascadeClassifier& nestedCascade,
                   double scale)
{
    vector<Rect> faces, faces2;
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
    double fx = 1 / scale;

    // Resize the Grayscale Image
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    // Detect faces of different sizes using cascade classifier
    cascade.detectMultiScale( smallImg, faces, 1.1,
                              2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    // Draw circles around the faces
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];

        if (r.x > 320)
        {
            if (timer > 0)
            {
                if (timer - unix_timestamp() <= -1)
                {
                    counter++;
                    cout << counter << endl;
                }
            }

        }

        if (r.x < 320)
        {
           if (counter > 0)
           {
               if (timer > 0)
               {
                   if (timer - unix_timestamp() <= -1)
                   {
                       counter--;
                       cout << counter << endl;
                   }
               }

           } else
           {
                counter = 0;
           }
        }

        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = Scalar(255, 0, 0); // Color for Drawing tool
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                       Point(cvRound((r.x + r.width-1)*scale),
                               cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );

        // Detection of eyes int the input image
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects, 1.1, 2,
                                        0|CASCADE_SCALE_IMAGE, Size(30, 30) );

        // Draw circles around eyes
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }

    Rect rect(0,0, 320,480);
    rectangle(img,rect,Scalar(0,255,255));
    Rect rect2(320, 0, 640,480);
    rectangle(img,rect2,Scalar(255,0,255));
    timer = unix_timestamp();
    // Show Processed Image with detected faces
    imshow( "Face Detection", img );
}

int main(int argc, char** argv) {

    CascadeClassifier cascadeUpper, cascadeLower;

    cascadeUpper.load("/home/developer/opencv/opencv/data/haarcascades/haarcascade_profileface.xml");
    cascadeLower.load("/home/developer/opencv/opencv/data/haarcascades/haarcascade_eye.xml");


    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        return -1;
    }

    Mat frame;

    for(;;)
    {
        cap >> frame;
        if (frame.empty()) break;

        Mat frame1 = frame.clone();
        detectAndDraw(frame1, cascadeUpper, cascadeLower, 1);

        if (waitKey(30) >= 0) break;
    }

    return 0;
}

