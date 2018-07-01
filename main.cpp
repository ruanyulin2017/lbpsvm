//#include"SVMTest.h"
#include "SVMTest.h"

int main(int argc, char *argv[])
{
    string root="/home/ruanyulin/Downloads/faceDatabase/gt_db/";
    SVMTest svmTest(root+"at.txt", // train filelist
                    root+"Test.txt", // test filelist
                    root+"Classifier.xml",  // classifier
                    root+"PredictResult_57.txt",  //predict result
                    SVM::C_SVC,  // svmType
                    SVM::LINEAR, // kernel
                    1,   // c
                    0,   // coef
                    0,   // degree
                    1,   // gamma
                    0,   // nu
                    0);  // p
    if(!svmTest.Initialize())
    {
        printf("initialize failed!\n");
        //LOG_ERROR_SVM_TEST("initialize failed!");
        return 0;
    }

    //svmTest.Train();
    //svmTest.Predict();




    cout<<"circle"<<endl;
    Mat dst;
    //dst.setTo(0);
    Mat src = imread("/home/ruanyulin/Desktop/rl.jpg",0);

    //namedWindow("gray",WINDOW_AUTOSIZE);
    //imshow("gray",src);
    LBP lbp;
    lbp.getCircularLBPFeatureOptimization(src,dst,1,8);

    namedWindow("circle",WINDOW_AUTOSIZE);
    imshow("circle",dst);


    waitKey(0);
    return 0;

}