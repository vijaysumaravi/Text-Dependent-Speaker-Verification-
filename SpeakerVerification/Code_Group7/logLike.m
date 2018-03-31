function likScores = logLike(featureDict,gmmMeanDict,gmmVarDict,gmmWeightDict,fileList1,fileList2)
%Calculates log of likelihood
%
% Inputs:   featureDict     feature set
%           gmmMeanDict     gmm means
%           gmmVarDict      gmm variances
%           gmmWeightDict   gmm weights
%           fileList1       first list of files for comparison
%           fileList2       second list of files for comparison
%
% Outputs:  likScores     	log-likelihood (average)

likScores = zeros(size(fileList1,1),1);

for i = 1:size(fileList1,1)
    likScores(i)=mean(gaussmixp(featureDict(fileList2{i}),...
        gmmMeanDict(fileList1{i}),gmmVarDict(fileList1{i}),...
        gmmWeightDict(fileList1{i})));   
end

% for i = 1:size(fileList1,1)
%     likScores(i+size(fileList1,1))=mean(gaussmixp(featureDict(fileList1{i}),...
%         gmmMeanDict(fileList2{i}),gmmVarDict(fileList2{i}),...
%         gmmWeightDict(fileList2{i})));   
% end

end