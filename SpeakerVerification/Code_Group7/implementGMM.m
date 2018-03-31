function [meanDict,varDict,weightDict,svDict] = implementGMM(featureDict,fileList,num_gaussian)
%Implements GMM, returns models, and means as supervectors in a dictionary
%
% Inputs:   featureDict     feature set
%           fileList        list of files for feature dataset
%           num_gaussian    number of Gaussians
%
% Outputs:  meanDict        dictionary of means
%           varDict         dictionary of variances
%           weightDict      dictionary of weights
%           svDict          dictionary of mean supervectors

num_features = size(featureDict(fileList{1}),2);
svDict = containers.Map;
meanDict = containers.Map;
varDict = containers.Map;
weightDict = containers.Map;
for i = 1:size(featureDict,1)
    [m,v,w]=gaussmix(featureDict(fileList{i}),[],[],num_gaussian);
    meanDict(fileList{i}) = m;
    varDict(fileList{i}) = v;
    weightDict(fileList{i}) = w;
    % Convert to supervector
    svDict(fileList{i}) = reshape(m,[1,num_gaussian*num_features]);
end

end