% Main program file

clear all
clc

% Parameters:
% Extract features
EXTRACT_FEATURES = false;                       % Extract features, else load feature data (bool)
GMM_FEATURES_OPT = [0 0 0];                     % Request features for GMM [LPC PLP GFCC]
KNN_FEATURES_OPT = [1 1 1];                     % Request features for KNN [LPC PLP GFCC]
SVM_FEATURES_OPT = [1 0 1];                     % Request features for SVM [LPC PLP GFCC]
NB_FEATURES_OPT = [0 0 0];                      % Request features for NB [LPC PLP GFCC]
V_THRES = 0.45;                                 % Voiced/unvoiced threshold
VOICED_ONLY = true;                             % Analyze only voiced frames (bool)
% Feature set to load for each model
GMM_FEATURE_SET = 'featuresCore.mat';
KNN_FEATURE_SET = 'featuresLPC_PLP_GFCC.mat';
SVM_FEATURE_SET = 'featuresLPC_GFCC.mat';
NB_FEATURE_SET = 'featuresCore.mat';
% GMM
IMPLEMENT_GMM = true;                           % Implement GMM-log likelihood (bool)
NUM_GAUSSIAN = 4;                               % Number of Gaussian mixtures
% k-NN
IMPLEMENT_KNN = true;                           % Implement k-NN (bool)
NUM_NEIGHBORS = 5;                              % Number of neighbors
% SVM
IMPLEMENT_SVM = true;                           % Implement SVM (bool)
% Naive-Bayes
IMPLEMENT_NB = true;                            % Implement Naive-Bayes (bool)
% Score Fusion
IMPLEMENT_SF = true;                            % Implement score fusion (bool)

% Define file lists
allList = 'allList.txt';
trainClean = 'trainCleanList.txt';
trainMulti = 'trainMultiList.txt';
testClean = 'testCleanList.txt'; 
testBabble = 'testBabbleList.txt';

% Define training and test lists
trainList = trainClean;
testCleanList = testClean;
testBabbleList = testBabble;

tic

fid = fopen(allList);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};

% Obtain feature dataset
disp('Obtaining feature dataset')
if EXTRACT_FEATURES
    % Extract features
    if IMPLEMENT_GMM
        gmmFeatureDict = extractFeatures(myFiles,V_THRES,GMM_FEATURES_OPT,VOICED_ONLY);
    end
    if IMPLEMENT_KNN
        knnFeatureDict = extractFeatures(myFiles,V_THRES,KNN_FEATURES_OPT,VOICED_ONLY);
    end
    if IMPLEMENT_SVM
        svmFeatureDict = extractFeatures(myFiles,V_THRES,SVM_FEATURES_OPT,VOICED_ONLY);
    end
    if IMPLEMENT_NB
        nbFeatureDict = extractFeatures(myFiles,V_THRES,NB_FEATURES_OPT,VOICED_ONLY);
    end
else
    % Load feature sets
    if IMPLEMENT_GMM
        load(GMM_FEATURE_SET);
        gmmFeatureDict = featureDict;
    end
    if IMPLEMENT_SVM
        load(SVM_FEATURE_SET);
        svmFeatureDict = featureDict;
    end
    if IMPLEMENT_KNN
        load(KNN_FEATURE_SET);
        knnFeatureDict = featureDict;
    end
    if IMPLEMENT_NB
        load(NB_FEATURE_SET);
        nbFeatureDict = featureDict;
    end
end

% Implement GMM
if IMPLEMENT_GMM
    disp('Implementing GMM')
    [gmmMeanDict,gmmVarDict,gmmWeightDict,gmmSvDict] = implementGMM(gmmFeatureDict,myFiles,NUM_GAUSSIAN);
end

% Compute feature mean and variance
if IMPLEMENT_KNN
    [knnMeanDict,knnVarDict] = computeMeanVar(knnFeatureDict,myFiles);
end
if IMPLEMENT_SVM
    [svmMeanDict,svmVarDict] = computeMeanVar(svmFeatureDict,myFiles);
end
if IMPLEMENT_NB
    [nbMeanDict,nbVarDict] = computeMeanVar(nbFeatureDict,myFiles);
end
    
% Training stage
disp('Training')
fid = fopen(trainList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
trainFileList1 = myData{1};
trainFileList2 = myData{2};
trainLabels = myData{3};

% Sample training dataset so that it is approximately balanced
[sampledTrainFileList1,sampledTrainFileList2,sampledTrainLabels] = ...
    sampleDataset(trainFileList1,trainFileList2,trainLabels);

% GMM: log likelihood
if IMPLEMENT_GMM
    disp('GMM')
    gmm_trainLikScores = logLike(gmmFeatureDict,gmmMeanDict,gmmVarDict,gmmWeightDict,...
        sampledTrainFileList1,sampledTrainFileList2);
    if ~IMPLEMENT_SF
        [gmm_trainPredict,accuracy,fpr,fnr,AUC,eer,gmm_threshold] = ...
            scoresAnalysis(gmm_trainLikScores,sampledTrainLabels);
        gmm_trainAnalysis = [accuracy,fpr,fnr,AUC,eer];
    end
end

% k-NN: absDiffMeanVar
if IMPLEMENT_KNN
    disp('k-NN')
    % Process input variables
    knnAbsDiffMean = absDiffArray(knnMeanDict,sampledTrainFileList1,sampledTrainFileList2);
    knnAbsDiffVar = absDiffArray(knnVarDict,sampledTrainFileList1,sampledTrainFileList2);
    knnAbsDiffMeanVar = [knnAbsDiffMean knnAbsDiffVar];
    % Training
    knn = fitcknn(knnAbsDiffMeanVar,sampledTrainLabels,'NumNeighbors',...
        NUM_NEIGHBORS,'Distance','cosine','Standardize',true);
    [knn_trainLabels,knn_trainScores,~] = predict(knn,knnAbsDiffMeanVar);
    if ~IMPLEMENT_SF
        [knn_trainPredict,accuracy,fpr,fnr,~,~,knn_threshold] = scoresAnalysis(knn_trainScores(:,2),...
            sampledTrainLabels,0.5);
        knn_trainAnalysis = [accuracy,fpr,fnr];
    end
end

% SVM: absDiffMeanVar
if IMPLEMENT_SVM
    disp('SVM')
    % Process input variables
    svmAbsDiffMean = absDiffArray(svmMeanDict,sampledTrainFileList1,sampledTrainFileList2);
    svmAbsDiffVar = absDiffArray(svmVarDict,sampledTrainFileList1,sampledTrainFileList2);
    svmAbsDiffMeanVar = [svmAbsDiffMean svmAbsDiffVar];
    % Training
    svm = fitrsvm(svmAbsDiffMeanVar,sampledTrainLabels,'Standardize',true);
    svm_trainScores = predict(svm,svmAbsDiffMeanVar);
    if ~IMPLEMENT_SF
        [svm_trainPredict,accuracy,fpr,fnr,~,~,svm_threshold] = scoresAnalysis(svm_trainScores,...
            sampledTrainLabels);
        svm_trainAnalysis = [accuracy,fpr,fnr];
    end
end

% Naive-Bayes: absDiffMeanVar
if IMPLEMENT_NB
    disp('Naive-Bayes')
    % Process input variables
    nbAbsDiffMean = absDiffArray(nbMeanDict,sampledTrainFileList1,sampledTrainFileList2);
    nbAbsDiffVar = absDiffArray(nbVarDict,sampledTrainFileList1,sampledTrainFileList2);
    nbAbsDiffMeanVar = [nbAbsDiffMean nbAbsDiffVar];
    % Training
    nb = fitcnb(nbAbsDiffMeanVar,sampledTrainLabels,'Distribution', 'kernel');
    [~,nb_trainScores,~] = predict(nb,nbAbsDiffMeanVar);
    if ~IMPLEMENT_SF
        [nb_trainPredict,accuracy,fpr,fnr,~,~,nb_threshold] = scoresAnalysis(nb_trainScores(:,2),...
            sampledTrainLabels);
        nb_trainAnalysis = [accuracy,fpr,fnr];
    end
end

% Score fusion
if IMPLEMENT_SF
    disp('Score Fusion')
    train_scores = [];
    if IMPLEMENT_GMM
        train_scores = [train_scores gmm_trainLikScores];
    end
    if IMPLEMENT_KNN
        train_scores = [train_scores knn_trainScores];
    end
    if IMPLEMENT_SVM
        train_scores = [train_scores svm_trainScores];
    end
    if IMPLEMENT_NB
        train_scores = [train_scores nb_trainScores];
    end
    regressor = fitrlinear(train_scores,sampledTrainLabels,'Learner',...
        'svm','Regularization','lasso');
    reg_trainScores = predict(regressor,train_scores);
    [reg_trainPredict,accuracy,fpr,fnr,~,~,reg_threshold]...
        = scoresAnalysis(reg_trainScores,sampledTrainLabels);
    reg_trainAnalysis = [accuracy,fpr,fnr];
end

for i = 1:2
    % Testing stage
    if i == 1
        testList = testCleanList;
    else
        testList = testBabbleList;
    end
    % Testing stage
    disp('Testing')
    fid = fopen(testList);
    myData = textscan(fid,'%s %s %f');
    fclose(fid);
    testFileList1 = myData{1};
    testFileList2 = myData{2};
    testLabels = myData{3};

    % GMM: log likelihood
    if IMPLEMENT_GMM
        disp('GMM')
        gmm_testLikScores = logLike(gmmFeatureDict,gmmMeanDict,gmmVarDict,gmmWeightDict,...
            testFileList1,testFileList2);
        if ~IMPLEMENT_SF
            [gmm_testPredict,accuracy,fpr,fnr,~,~,~] = scoresAnalysis(gmm_testLikScores,...
                testLabels,gmm_threshold);
            gmm_testAnalysis = [accuracy,fpr,fnr];
        end
    end

    % k-NN: absDiffMeanVar
    if IMPLEMENT_KNN
        disp('k-NN')
        % Process input variables
        absDiffMean = absDiffArray(knnMeanDict,testFileList1,testFileList2);
        absDiffVar = absDiffArray(knnVarDict,testFileList1,testFileList2);
        absDiffMeanVar = [absDiffMean absDiffVar];
        % Testing
        [~,knn_testScores,~] = predict(knn,absDiffMeanVar);
        if ~IMPLEMENT_SF
            [knn_testPredict,accuracy,fpr,fnr,~,~,~] = scoresAnalysis(knn_testScores(:,2),...
                testLabels,knn_threshold);
            knn_testAnalysis = [accuracy,fpr,fnr];
        end
    end

    % SVM: absDiffMeanVar
    if IMPLEMENT_SVM
        disp('SVM')
        % Process input variables
        absDiffMean = absDiffArray(svmMeanDict,testFileList1,testFileList2);
        absDiffVar = absDiffArray(svmVarDict,testFileList1,testFileList2);
        absDiffMeanVar = [absDiffMean absDiffVar];
        % Testing
        svm_testScores = predict(svm,absDiffMeanVar);
        if ~IMPLEMENT_SF
            [svm_testPredict,accuracy,fpr,fnr,~,~,~] = scoresAnalysis(svm_testScores,...
                testLabels,svm_threshold);
            svm_testAnalysis = [accuracy,fpr,fnr];
        end
    end

    % Naive-Bayes: absDiffMeanVar
    if IMPLEMENT_NB
        disp('Naive-Bayes')
        % Process input variables
        absDiffMean = absDiffArray(nbMeanDict,testFileList1,testFileList2);
        absDiffVar = absDiffArray(nbVarDict,testFileList1,testFileList2);
        absDiffMeanVar = [absDiffMean absDiffVar];
        % Testing
        [~,nb_testScores,~] = predict(nb,absDiffMeanVar);
        if ~IMPLEMENT_SF
            [nb_testPredict,accuracy,fpr,fnr,~,~,~] = scoresAnalysis(nb_testScores(:,2),...
                testLabels,nb_threshold);
            nb_testAnalysis = [accuracy,fpr,fnr];
        end
    end

    % Score fusion
    if IMPLEMENT_SF
        disp('Score Fusion')
        test_scores = [];
        if IMPLEMENT_GMM
            test_scores = [test_scores gmm_testLikScores];
        end
        if IMPLEMENT_KNN
            test_scores = [test_scores knn_testScores];
        end
        if IMPLEMENT_SVM
            test_scores = [test_scores svm_testScores];
        end
        if IMPLEMENT_NB
            test_scores = [test_scores nb_testScores];
        end
        reg_testScores = predict(regressor,test_scores);
        [reg_testPredict,accuracy,fpr,fnr,~,~,reg_threshold] = scoresAnalysis(...
            reg_testScores,testLabels,reg_threshold);
        reg_testAnalysis = [accuracy,fpr,fnr];
    end
end

toc