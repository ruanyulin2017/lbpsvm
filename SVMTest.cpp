//
// Created by ruanyulin on 18-4-13.
//

#include"SVMTest.h"

SVMTest::SVMTest(const string &_trainDataFileList,
                 const string &_testDataFileList,
                 const string &_svmModelFilePath,
                 const string &_predictResultFilePath,
                 SVM::Types svmType, // See SVM::Types. Default value is SVM::C_SVC.
                 SVM::KernelTypes kernel,
                 double c, // For SVM::C_SVC, SVM::EPS_SVR or SVM::NU_SVR. Default value is 0.
                 double coef,  // For SVM::POLY or SVM::SIGMOID. Default value is 0.
                 double degree, // For SVM::POLY. Default value is 0.
                 double gamma, // For SVM::POLY, SVM::RBF, SVM::SIGMOID or SVM::CHI2. Default value is 1.
                 double nu,  // For SVM::NU_SVC, SVM::ONE_CLASS or SVM::NU_SVR. Default value is 0.
                 double p // For SVM::EPS_SVR. Default value is 0.
):
        trainDataFileList(_trainDataFileList),
        testDataFileList(_testDataFileList),
        svmModelFilePath(_svmModelFilePath),
        predictResultFilePath(_predictResultFilePath)
{
    // set svm param
    svm = SVM::create();
    svm->setC(c);
    svm->setCoef0(coef);
    svm->setDegree(degree);
    svm->setGamma(gamma);
    svm->setKernel(kernel);
    svm->setNu(nu);
    svm->setP(p);
    svm->setType(svmType);

    //svm->setTermCriteria(TermCriteria(TermCriteria::EPS, 1000, FLT_EPSILON)); // based on accuracy
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6)); // based on the maximum number of iterations
}

bool SVMTest::Initialize()
{
    // initialize log
    //InitializeLog("SVMTest");

    return true;

}

SVMTest::~SVMTest()
{
}

void SVMTest::Train()
{

    // 读入训练样本图片路径和类别
    std::vector<string> imagePaths;
    std::vector<int> imageClasses;
    string line;
    //string path = "/home/ruanyulin/Desktop/images/";
    std::ifstream trainingData(trainDataFileList,ios::out);
    while (getline(trainingData, line))
    {
        if(line.empty())
            continue;

        //line = path+line;
        stringstream stream(line);
        string imagePath,imageClass;
        stream>>imagePath;
        stream>>imageClass;

        imagePaths.push_back(imagePath);
        imageClasses.push_back(atoi(imageClass.c_str()));
    }
    trainingData.close();

    printf("%d\n",imagePaths.size());

    // extract feature
    Mat featureVectorsOfSample;
    Mat classOfSample;
    Mat srcImage;
    Mat featureVector;
    printf("get feature...\n");
    LOG_INFO_SVM_TEST("get feature...");
    Mat resizess;
    for(int i=0;i<=imagePaths.size()-1;++i)
    {
        srcImage = imread(imagePaths[i],0);



        //printf("depth:%d\n",srcImage.depth());
        //waitKey(4000);
        if(srcImage.empty()||srcImage.depth()!=CV_8U)
        {
            if (srcImage.empty())
            {
                printf("empty\n");
            }
            printf("%s srcImage.empty()||srcImage.depth()!=CV_8U!\n",imagePaths[i].c_str());
            LOG_ERROR_SVM_TEST("%s srcImage.empty()||srcImage.depth()!=CV_8U!",imagePaths[i].c_str());
            continue;
        }
        /*if (srcImage.cols!=100||srcImage.rows!=114)
        {
            resize(srcImage,resizess,Size(100,114),INTER_LINEAR);
            namedWindow("res",WINDOW_AUTOSIZE);
            //imshow("res",srcImage);
            //waitKey(555555);
            srcImage=resizess;
        }*/
        // extract feature

        //lbp.ComputeLBPFeatureVector_Rotation_Uniform(srcImage,Size(CELL_SIZE,CELL_SIZE),featureVector);
        lbp.ComputeLBPFeatureVector_256(srcImage,Size(CELL_SIZE,CELL_SIZE),featureVector);
        //lbp.ComputeLBPFeatureVector_Uniform(srcImage,Size(CELL_SIZE,CELL_SIZE),featureVector);

        if(featureVector.empty())
            continue;

        featureVectorsOfSample.push_back(featureVector);
        classOfSample.push_back( imageClasses[i]);

        printf("feature getting： %f \n ",(i+1)*100.0/imagePaths.size());
        LOG_INFO_SVM_TEST("get feature... %f",(i+1)*100.0/imagePaths.size());
    }

    printf("get feature done!\n");
    LOG_INFO_SVM_TEST("get feature done!");

    // train
    printf("training...\n");
    LOG_INFO_SVM_TEST("training...");
    double time1, time2;
    time1 = getTickCount();
    svm->train(featureVectorsOfSample, ROW_SAMPLE, classOfSample);
    //svm->trainAuto(featureVectorsOfSample,ROW_SAMPLE,classOfSample);
    time2 = getTickCount();
    printf("training time for %dsamples:%f\n", imagePaths.size(),(time2 - time1)*1000. / getTickFrequency());
    LOG_INFO_SVM_TEST("training time:%f", (time2 - time1)*1000. / getTickFrequency());
    printf("training done\n");
    LOG_INFO_SVM_TEST("training done!");

    // save model
    //svm->save(svmModelFilePath);

}

void SVMTest::Predict()
{
    // predict
    Mat resizess;
    std::vector<string> testImagePaths;
    std::vector<int> testImageClasses;
    string line;
    string path = "/home/ruanyulin/Desktop/images/";
    double time1,time2;
    std::ifstream testData(testDataFileList,ios::out);
    time1 = getTickCount();
    while (getline(testData, line))
    {
        if(line.empty())
            continue;

        //line = path+line;
        stringstream stream(line);
        string imagePath,imageClass;
        stream>>imagePath;
        stream>>imageClass;

        testImagePaths.push_back(imagePath);
        testImageClasses.push_back(atoi(imageClass.c_str()));

    }
    testData.close();

    printf("predict %d \n",testImagePaths.size());
    printf("predicting...\n");
    LOG_INFO_SVM_TEST("predicting...");

    int numberOfRight_0=0;
    int numberOfError_0=0;
    //int numberOfRight_1=0;
    //int numberOfError_1=0;

    std::ofstream fileOfPredictResult(predictResultFilePath,ios::out); //最后识别的结果
    double sum_Predict=0,sum_ExtractFeature=0;
    char line2[256]={0};
    Mat srcImage;
    Mat resizes;
    for (int i = 0; i < testImagePaths.size() ; ++i)
    {
        srcImage = imread(testImagePaths[i], 0);
        //printf("depth: %d \n",srcImage.depth());
        //printf("width:%d \n",srcImage.rows);
        //printf("height:%d \n",srcImage.cols);
        //printf("channel:%d \n",srcImage.channels());
        /*if(srcImage.rows!=300||srcImage.cols!=300)
        {

            resize(srcImage,resizes,Size(300,300),INTER_LINEAR);
            srcImage = resizes;
        }*/
        /*if(srcImage.channels()!=1)
        {
            Mat gray_image;
            resize(srcImage,gray_image,Size(112,92),INTER_LINEAR);


            cvtColor(gray_image,srcImage,CV_BGR2GRAY);
            //srcImage = gray_image;
            //namedWindow("gray",WINDOW_AUTOSIZE);
            //imshow("gray",srcImage);
            //std::cout<<"gray"<<endl;
            //printf("setchannel:%d \n",srcImage.channels());
            //waitKey(0);
        }*/
        /*if (srcImage.cols!=100||srcImage.rows!=114)
        {
            resize(srcImage,resizess,Size(100,114),INTER_LINEAR);
            srcImage=resizess;
        }*/

        if(srcImage.empty()||srcImage.depth()!=CV_8U)
        {
            printf("%s srcImage.empty()||srcImage.depth()!=CV_8U!\n",testImagePaths[i].c_str());
            LOG_ERROR_SVM_TEST("%s srcImage.empty()||srcImage.depth()!=CV_8U!",testImagePaths[i].c_str());
            continue;
        }
        //printf("depth: %d \n",srcImage.depth());
        // extract feature
        double time1_ExtractFeature = getTickCount();
        Mat featureVectorOfTestImage;
        //lbp.ComputeLBPFeatureVector_Rotation_Uniform(srcImage,Size(CELL_SIZE,CELL_SIZE),featureVectorOfTestImage);
        lbp.ComputeLBPFeatureVector_256(srcImage,Size(CELL_SIZE,CELL_SIZE),featureVectorOfTestImage);
        //lbp.ComputeLBPFeatureVector_Uniform(srcImage,Size(CELL_SIZE,CELL_SIZE),featureVectorOfTestImage);

        //printf("extract feature...\n");
        if(featureVectorOfTestImage.empty())
            continue;
        double time2_ExtractFeature = getTickCount();
        sum_ExtractFeature+=(time2_ExtractFeature - time1_ExtractFeature) * 1000 / getTickFrequency();

        //对测试图片进行分类并写入文件
        double time1_Predict = getTickCount();
        int predictResult = svm->predict(featureVectorOfTestImage);
        printf("predict result：%d",predictResult);
        string result_path;
        result_path ="/home/ruanyulin/Downloads/face/s";


        //printf("result_path: %s \n",result_path);
        Mat result;
        //result = imread(result_path,-1);
        //namedWindow("result",WINDOW_AUTOSIZE);
        //imshow("result",result);
        double time2_Predict = getTickCount();
        sum_Predict += (time2_Predict - time1_Predict) * 1000 / getTickFrequency();

        sprintf(line2, "%s %d\n", testImagePaths[i].c_str(), predictResult);
        fileOfPredictResult << line2;
        LOG_INFO_SVM_TEST("%s %d", testImagePaths[i].c_str(), predictResult);

        // 0
        if(testImageClasses[i]==predictResult)
        {
            ++numberOfRight_0;
            printf("\n");
        }
        if(testImageClasses[i]!=predictResult)
        {
            ++numberOfError_0;
            printf(" error:%d\n",testImageClasses[i]);
            //printf("错误\n");
        }

        // 1
        if((testImageClasses[i]==1)&&(predictResult==1))
        {
            //++numberOfRight_1;
        }
        if((testImageClasses[i]==1)&&(predictResult!=1))
        {
            //++numberOfError_1;
        }

        //printf("predicting  %f \n", 100.0*(i+1)/testImagePaths.size());
    }
    time2 = getTickCount();
    printf("predict time for %dsamples:%f\n", testImagePaths.size(),(time2 - time1)*1000. / getTickFrequency());
    svm->clear();
    printf("predicting done!\n");
    LOG_INFO_SVM_TEST("predicting done!");


    //printf("extract feature time：%f\n", sum_ExtractFeature / testImagePaths.size());
    LOG_INFO_SVM_TEST("extract feature time：%f", sum_ExtractFeature / testImagePaths.size());
    sprintf(line2, "extract feature time：%f\n", sum_ExtractFeature / testImagePaths.size());
    fileOfPredictResult << line2;

    //printf("predict time：%f\n", sum_Predict / testImagePaths.size());
    LOG_INFO_SVM_TEST("predict time：%f", sum_Predict / testImagePaths.size());
    sprintf(line2, "predict time：%f\n", sum_Predict / testImagePaths.size());
    fileOfPredictResult << line2;



    // 0
    double accuracy_0=(100.0*(numberOfRight_0)) / (numberOfError_0+numberOfRight_0);
    //printf("0：%f\n",accuracy_0);
    LOG_INFO_SVM_TEST("0：%f",accuracy_0);
    sprintf(line2, "0：%f\n", accuracy_0);
    fileOfPredictResult << line2;

    double accuracy;
    accuracy = (100.0*(numberOfRight_0))/testImagePaths.size();
    printf("Accuracy:%f \n",accuracy);
    double error;
    error = (100.0*(numberOfError_0))/testImagePaths.size();
    printf("error rate:%f \n:",error);
    printf("wrong number:%d \n",numberOfError_0);
    // 1
    //double accuracy_1=(100.0*numberOfRight_1) /(numberOfError_1+numberOfRight_1);
    //printf("1：%f\n",accuracy_1);
    LOG_INFO_SVM_TEST("1：%f", accuracy_1);
    //sprintf(line2, "1：%f\n",accuracy_1);
    fileOfPredictResult << line2;

    // accuracy
    //double accuracy_All=(100.0*(numberOfRight_1+numberOfRight_0)) /(numberOfError_0+numberOfRight_0+numberOfError_1+numberOfRight_1);
    //printf("accuracy：%f\n", accuracy_All);
    LOG_INFO_SVM_TEST("accuracy:%f", accuracy_All);
    //sprintf(line2, "accuracy:%f\n", accuracy_All);
    fileOfPredictResult << line2;

    fileOfPredictResult.close();

}
