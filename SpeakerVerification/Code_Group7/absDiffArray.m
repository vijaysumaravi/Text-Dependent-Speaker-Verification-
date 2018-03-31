function absDiff = absDiffArray(featureDict,fileList1,fileList2)
%Implements GMM, returns models, and means as supervectors in a dictionary
%
% Inputs:   featureDict     feature set
%           fileList1       first list of files for comparison
%           fileList2       second list of files for comparison
%
% Outputs:  absDiff         returns absolute difference in a 2D array

absDiff = zeros(size(fileList1,1),size(featureDict(fileList1{1}),2));

for i = 1:size(fileList1,1)
    A = featureDict(fileList1{i});
    B = featureDict(fileList2{i});
    
    absDiff(i,:) = abs(A-B);
end

end