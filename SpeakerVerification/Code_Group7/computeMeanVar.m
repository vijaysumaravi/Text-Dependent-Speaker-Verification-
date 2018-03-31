function [meanDict,varDict] = computeMeanVar(featureDict,fileList)
%Implements GMM, returns models, and means as supervectors in a dictionary
%
% Inputs:   featureDict     feature set
%           fileList        list of files for feature dataset
%
% Outputs:  meanDict        dictionary of feature means
%           varDict         dictionary of feature variances

meanDict = containers.Map;
varDict = containers.Map;
for i = 1:size(featureDict,1)
    meanDict(fileList{i}) = mean(featureDict(fileList{i}),1);
    varDict(fileList{i}) = var(featureDict(fileList{i}),0,1);
end

end