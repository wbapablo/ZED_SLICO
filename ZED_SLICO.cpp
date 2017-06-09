
#include "SLIC.h"
//Math





#include <iostream>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <iomanip>
#include <stddef.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


//#include <cuda.h>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>


#include <sstream>

#include <iostream>


using namespace cv;
//main  function

int main(int argc, char **argv) {

    if (argc > 3) {
        std::cout << "Only the path of a SVO or a InitParams file can be passed in arg." << std::endl;
        return -1;
    }

    // quick check of arguments
    bool readSVO = false;
    std::string SVOName;
    bool loadParams = false;
    std::string ParamsName;
    if (argc > 1) {
        std::string _arg;
        for (int i = 1; i < argc; i++) {
            _arg = argv[i];
            if (_arg.find(".svo") != std::string::npos) { // if a SVO is given we save its name
                readSVO = true;
                SVOName = _arg;
            }
            if (_arg.find(".ZEDinitParam") != std::string::npos) { // if a parameters file is given we save its name
                loadParams = true;
                ParamsName = _arg;
            }
        }
    }

    sl::zed::Camera* zed;

    if (!readSVO) // Use in Live Mode
        zed = new sl::zed::Camera(sl::zed::VGA);

    else // Use in SVO playback mode
        zed = new sl::zed::Camera(SVOName);

    // define a struct of parameters for the initialization
    sl::zed::InitParams params;
    //zed->setFPS(100);

   // if (loadParams)// a file is given in argument, we load it
   //     params.load(ParamsName);


    //sl::zed::ERRCODE err =
    zed->init(params);
    //std::cout << "Error code : " << sl::zed::errcode2str(err) << std::endl;
    /*if (err != sl::zed::SUCCESS) {// Quit if an error occurred
        delete zed;
        return 1;
    }*/

    // Save the initialization parameters
    params.save("MyParam");

    char key = ' ';
    int width = 640/2;
    int height = 480/2;

	int col_am=10;
	int row_am=8;
	int cucel;
	int bellow;
	int right;
	float color_d;
	float d_l, d_a, d_b;

	int group=1;


    cv::Size DisplaySize(width, height);
    cv::Mat dispDisplay(DisplaySize, CV_8UC3);


    cv::Mat disp(height, width, CV_8UC3);
    Mat img_seg(DisplaySize,CV_8UC3);
    Mat Connection_Matrix(8,10,CV_8UC2);


    int k = 80; // Desired number of superpixels.
    int sp_visit[k];
    int sp_group[k];
   	double m =10; // Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10

   	int numlabels = 0;
   	string filename("");
   	string savepath("");
    	//SEEDS and Colours

   	vector<double> kls;
   	vector<double> kas;
  	vector<double> kbs;
   	vector<double> kxs;
   	vector<double> kys;
   	Vec4b intensity;
    	//----------------------------------
    	// Perform SLIC on the image buffer
    	//----------------------------------
   	SLIC segment;
   	unsigned int *pbuff = new unsigned int[width*height];
   	int *klabels = new int[width*height];


    sl::zed::SENSING_MODE dm_type = sl::zed::STANDARD;

    zed->setFPS(60);

    zed->grab(dm_type,0,0,0);

    cv::namedWindow("VIEW", cv::WINDOW_AUTOSIZE);
   // cv::namedWindow("VIEW2", cv::WINDOW_AUTOSIZE);

    //loop until 'q' is pressed
    while (key != 'q') {


        // Get frames and launch the computation
        zed->grab(dm_type,0,0,0);
        slMat2cvMat(zed->retrieveImage(sl::zed::LEFT)).copyTo(disp);

        resize(disp, dispDisplay, DisplaySize);

        //cout<<"width R: "<< dispDisplay.cols << "  height R:" << dispDisplay.rows << endl;
        //cout<<"width R: "<< img_seg[1].dims << "  height R:" << img_seg.rows << endl;


        unsigned int blue, green, red;
        for (int i = 0; i < height; i++) {
        	for (int j = 0; j < width; j++) {
        		intensity = dispDisplay.at<Vec4b>(i, j);
       			blue = intensity.val[0];
       			green = intensity.val[1];
       			red = intensity.val[2];
       			pbuff[i*width+j] = (red << 16) + (green << 8) + blue;

       		}
       	}

        segment.PerformSLICO_ForGivenK(pbuff, width, height, klabels, numlabels, k, m, kls, kas, kbs, kxs, kys);
        segment.DrawContoursAroundSegments(pbuff, klabels, width, height, 0xff0000);
    	for (int i = 0; i < height; i++) {
    		for (int j = 0; j < width; j++) {
    			uchar *color = (uchar*)(pbuff+i*width+j);
    			img_seg.at<Vec3b>(i, j)[0] = color[0];
    			img_seg.at<Vec3b>(i, j)[1] = color[1];
    			img_seg.at<Vec3b>(i, j)[2] = color[2];
    		}
    	}

    	cout << "Super pixels: " << numlabels << endl;

    	//
    	if(numlabels==80){
    		//Llenar matriz con coordenadas de centroides
    		for (int r_i =0; r_i < col_am;r_i++){
    			for(int c_i=0; c_i < row_am; c_i++) {
    				cucel=r_i*8 + c_i;
    				Connection_Matrix.at<Vec2b>(r_i,c_i)[0] = kxs[cucel];
    				Connection_Matrix.at<Vec2b>(r_i,c_i)[1] = kys[cucel];
    				sp_group[cucel]=0;
					sp_visit[cucel]=0;

    			}
    		}
    		sp_visit[0]=1;
    		//checar adyacencia y además similitud
    		for (int r_i =0; r_i < col_am;r_i++){
    			for(int c_i=0; c_i < row_am; c_i++) {
    				//Checar si es una celda que se debe visitar
    				if( sp_visit[cucel]==1){

    					sp_group[cucel]=group;
    					//Verificar por debajo
    					bellow= 8*(r_i+1)+c_i;
    					d_l=kls[cucel]-kls[bellow];
    					d_a=kas[cucel]-kas[bellow];
    					d_b=kbs[cucel]-kbs[bellow];
    					color_d=sqrt(pow(d_l,2) + pow(d_a,2) + pow(d_b,2));
    					if (color_d<35){
    						sp_group=group;
    						sp_visit=1;
    					}

    				}
    			}
    		}


    	}





    	imshow("VIEW", dispDisplay);
    	imshow("VIEW2",img_seg);
    	key = waitKey(5);
    }

    if (pbuff) delete [] pbuff;
   	if (klabels) delete [] klabels;

    delete zed;
    return 0;
}
