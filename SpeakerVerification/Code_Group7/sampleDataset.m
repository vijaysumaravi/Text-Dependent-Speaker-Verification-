function [sampledFileList1,sampledFileList2,sampledLabels] = sampleDataset(fileList1,fileList2,labels)
%Implements GMM, returns models, and means as supervectors in a dictionary
%
% Inputs:   fileList1           first list of files for comparison
%           fileList2           second list of files for comparison
%           labels              both files by same speaker (binary)
%
% Outputs:  sampledFileList1    sampled first list of file
%           sampledFileList2    sampled second list of file
%           sampledLabels       sampled labels

sampledFileList1 = [];
sampledFileList2 = [];
sampledLabels = [];

for i = 1:size(fileList1,1)   
    A = {fileList1{i}};
    B = {fileList2{i}};
    label = labels(i);
    if label == 1
        sampledFileList1 = [sampledFileList1;A;A];
        sampledFileList2 = [sampledFileList2;B;B];
        sampledLabels = [sampledLabels;label;label];
    else
        if rand <= 2.5*size(labels(labels==1))/size(labels(labels==0))
            sampledFileList1 = [sampledFileList1;A];
            sampledFileList2 = [sampledFileList2;B];
            sampledLabels = [sampledLabels;label];
        end
    end

end


end