function kld = kldGMM(gmmMeanDict,gmmVarDict,gmmWeightDict,fileList1,fileList2)
%Implements GMM, returns models, and means as supervectors in a dictionary
%
% Inputs:   gmmDict         dictionary of means, variances, weights
%           fileList1       first list of files for comparison
%           fileList2       second list of files for comparison
%
% Outputs:  kld         	KL divergence

kld = zeros(size(fileList1,1),1);

for i = 1:size(fileList1,1)   
    A = fileList1{i};
    B = fileList2{i};
    [kld(i),~] = gaussmixk(gmmMeanDict(A),gmmVarDict(A),gmmWeightDict(A),...
        gmmMeanDict(B),gmmVarDict(B),gmmWeightDict(B));
end

% for i = 1:size(fileList1,1)   
%     A = fileList2{i};
%     B = fileList1{i};
%     [kld(i+size(fileList1,1)),~] = gaussmixk(gmmMeanDict(A),gmmVarDict(A),gmmWeightDict(A),...
%         gmmMeanDict(B),gmmVarDict(B),gmmWeightDict(B));
% end

end