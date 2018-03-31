% Main program file

clear all
clc

% Set to false to run official test set
DEBUG_MODE = false;

% Define test list
testList = 'trialsEvaluation.txt';

% Load trained models
load('KNN_CLEAN.mat');
knn_clean = knn;

load('SVM_CLEAN.mat');
svm_clean = svm;

load('NB_CLEAN.mat');
nb_clean = nb;

load('REG_CLEAN.mat');
reg_clean = regressor;

load('REG_THRESHOLD_CLEAN.mat');
reg_threshold_clean = reg_threshold;

load('KNN_NOISY.mat');
knn_noisy = knn;

load('SVM_NOISY.mat');
svm_noisy = svm;

load('NB_NOISY.mat');
nb_noisy = nb;

load('REG_NOISY.mat');
reg_noisy = regressor;

load('REG_THRESHOLD_NOISY.mat');
reg_threshold_noisy = reg_threshold;

% General parameters
V_THRES = 0.45;                             % Voiced/unvoiced threshold
VOICED_ONLY = true;                         % Analyze only voiced frames (bool)

% Clean model parameters
GMM_FEATURES_OPT_CLEAN = [0 0 0];           % Request features for GMM [LPC PLP GFCC]
NUM_GAUSSIAN_CLEAN = 4;
KNN_FEATURES_OPT_CLEAN = [1 1 1];           % Request features for KNN [LPC PLP GFCC]
SVM_FEATURES_OPT_CLEAN = [1 0 1];           % Request features for SVM [LPC PLP GFCC]
NB_FEATURES_OPT_CLEAN = [0 0 0];            % Request features for NB [LPC PLP GFCC]
% Noisy model parameters
GMM_FEATURES_OPT_NOISY = [1 0 1];           % Request features for GMM [LPC PLP GFCC]
NUM_GAUSSIAN_NOISY = 8;
KNN_FEATURES_OPT_NOISY = [0 0 1];           % Request features for KNN [LPC PLP GFCC]
SVM_FEATURES_OPT_NOISY = [1 1 1];           % Request features for SVM [LPC PLP GFCC]
NB_FEATURES_OPT_NOISY = [0 1 1];            % Request features for NB [LPC PLP GFCC]

% Testing stage
disp('Testing')
if DEBUG_MODE
    fid = fopen('testCleanList.txt');
    myData = textscan(fid,'%s %s %f');
    fclose(fid);
    testFileList1 = myData{1};
    fid = fopen('testBabbleList.txt');
    myData = textscan(fid,'%s %s %f');
    fclose(fid);
    testFileList2 = myData{2};
    myFiles = unique([testFileList1; testFileList2]);
    testLabels = myData{3};
else
    fid = fopen(testList);
    myData = textscan(fid,'%s %s');
    fclose(fid);
    testFileList1 = myData{1};
    testFileList2 = myData{2};
    myFiles = unique([testFileList1; testFileList2]);
end

% Extract features
disp('Extracting Features')
gmmFeatureDict_clean = extractFeatures(myFiles,V_THRES,GMM_FEATURES_OPT_CLEAN,VOICED_ONLY);
knnFeatureDict_clean = extractFeatures(myFiles,V_THRES,KNN_FEATURES_OPT_CLEAN,VOICED_ONLY);
svmFeatureDict_clean = extractFeatures(myFiles,V_THRES,SVM_FEATURES_OPT_CLEAN,VOICED_ONLY);
nbFeatureDict_clean = extractFeatures(myFiles,V_THRES,NB_FEATURES_OPT_CLEAN,VOICED_ONLY);

gmmFeatureDict_noisy = extractFeatures(myFiles,V_THRES,GMM_FEATURES_OPT_NOISY,VOICED_ONLY);
knnFeatureDict_noisy = extractFeatures(myFiles,V_THRES,KNN_FEATURES_OPT_NOISY,VOICED_ONLY);
svmFeatureDict_noisy = extractFeatures(myFiles,V_THRES,SVM_FEATURES_OPT_NOISY,VOICED_ONLY);
nbFeatureDict_noisy = extractFeatures(myFiles,V_THRES,NB_FEATURES_OPT_NOISY,VOICED_ONLY);

predictedLabels = zeros(size(testFileList1,1),1);
predictedScores = zeros(size(testFileList1,1),1);

for i = 1:size(testFileList1,1)
    i
    
    if ~detectNoise(testFileList1(i),testFileList2(i))
        gmmFeatureDict = gmmFeatureDict_clean;
        knnFeatureDict = knnFeatureDict_clean;
        svmFeatureDict = svmFeatureDict_clean;
        nbFeatureDict = nbFeatureDict_clean;
        knn = knn_clean;
        svm = svm_clean;
        nb = nb_clean;
        sf = reg_clean;
        reg_threshold = reg_threshold_clean;
        NUM_GAUSSIAN = NUM_GAUSSIAN_CLEAN;
    else
        gmmFeatureDict = gmmFeatureDict_noisy;
        knnFeatureDict = knnFeatureDict_noisy;
        svmFeatureDict = svmFeatureDict_noisy;
        nbFeatureDict = nbFeatureDict_noisy;
        knn = knn_noisy;
        svm = svm_noisy;
        nb = nb_noisy;
        sf = reg_noisy;
        reg_threshold = reg_threshold_noisy;
        NUM_GAUSSIAN = NUM_GAUSSIAN_NOISY;
    end
    
    % Compute feature mean and variance

    [knnMeanDict,knnVarDict] = computeMeanVar(knnFeatureDict,myFiles);
    [svmMeanDict,svmVarDict] = computeMeanVar(svmFeatureDict,myFiles);
    [nbMeanDict,nbVarDict] = computeMeanVar(nbFeatureDict,myFiles);
    
    % GMM: log likelihood
    [gmmMeanDict,gmmVarDict,gmmWeightDict,gmmSvDict] = implementGMM(gmmFeatureDict,myFiles,NUM_GAUSSIAN);
    gmm_testLikScores = logLike(gmmFeatureDict,gmmMeanDict,gmmVarDict,gmmWeightDict,...
        testFileList1(i),testFileList2(i));
    
    % k-NN: absDiffMeanVar
    % Process input variables
    absDiffMean = absDiffArray(knnMeanDict,testFileList1(i),testFileList2(i));
    absDiffVar = absDiffArray(knnVarDict,testFileList1(i),testFileList2(i));
    absDiffMeanVar = [absDiffMean absDiffVar];
    % Testing
    [~,knn_testScores,~] = predict(knn,absDiffMeanVar);
    
    % SVM: absDiffMeanVar
    % Process input variables
    absDiffMean = absDiffArray(svmMeanDict,testFileList1(i),testFileList2(i));
    absDiffVar = absDiffArray(svmVarDict,testFileList1(i),testFileList2(i));
    absDiffMeanVar = [absDiffMean absDiffVar];
    % Testing
    svm_testScores = predict(svm,absDiffMeanVar);
    
    % Naive-Bayes: absDiffMeanVar
    % Process input variables
    absDiffMean = absDiffArray(nbMeanDict,testFileList1(i),testFileList2(i));
    absDiffVar = absDiffArray(nbVarDict,testFileList1(i),testFileList2(i));
    absDiffMeanVar = [absDiffMean absDiffVar];
    % Testing
    [~,nb_testScores,~] = predict(nb,absDiffMeanVar);
    
    % Score fusion
    test_scores = [gmm_testLikScores knn_testScores svm_testScores nb_testScores];
    reg_testScores = predict(regressor,test_scores);
    if DEBUG_MODE
        [reg_testPredict,accuracy,fpr,fnr,~,~,reg_threshold] = scoresAnalysis(...
            reg_testScores,testLabels(i),reg_threshold);
    else
        if reg_testScores > reg_threshold
            reg_testPredict = 1;
        else
            reg_testPredict = 0;
        end
    end
    predictedLabels(i) = reg_testPredict;
    predictedScores(i) = reg_testScores;
end

if DEBUG_MODE
    tp = 0;
    tn = 0;
    fp = 0;
    fn = 0;

    for i = 1:size(testLabels,1)
        if predictedLabels(i)==1 && testLabels(i)==0
            fp = fp + 1;
        elseif predictedLabels(i)==0 && testLabels(i)==1
            fn = fn + 1;
        elseif predictedLabels(i)==0 && testLabels(i)==0
            tn = tn + 1;
        else
            tp = tp + 1;
        end
    end

    disp('Confusion Matrix')
    [tn fp;fn tp]
    accuracy = (tp+tn)/size(testLabels,1);
    fpr = fp/size(testLabels(testLabels(:)==0),1);
    fnr = fn/size(testLabels(testLabels(:)==1),1);
    disp('Accuracy FPR FNR')
    [accuracy fpr fnr]
else
    fid = fopen('scores.txt','w');
    formatSpec = '%s %s %d \r\n';
    [nrows,ncols] = size(x);
    for row = 1:nrows
        fprintf(fid,formatSpec,x{row,:});
    end
    fclose(fid);   
end