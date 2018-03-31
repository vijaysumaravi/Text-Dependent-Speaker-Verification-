function [predict,accuracy,fpr,fnr,auc,eer,threshold] = scoresAnalysis(scores,labels,threshold)
%Provides results analysis and returns threshold for eer
%
% Inputs:   scores          raw score
%           labels          ground truth labels
%           threshold       threshold for 1 or 0
%                           (use eer to determine if
%                           not provuded)
%
% Outputs:  accuracy        accuracy
%           fpr             false positive rate
%           fnr             false negative rate
%           auc             area under ROC
%           eer             equal error rate
%           threshold       optimal threshold based on eer


if nargin < 3
    [X,Y,T,auc] = perfcurve(labels,scores,1);
%     figure;
%     plot(X,Y)
    
    abs_diff = abs(X-1+Y);
    [eer,eer_index] = min(abs_diff(2:size(X)-1));
    threshold = T(eer_index);
else
    auc = 0;
    eer = 0;
end

predict = zeros(size(labels,1),1);
tp = 0;
tn = 0;
fp = 0;
fn = 0;

for i = 1:size(labels,1)
    if scores(i) > threshold
        predict(i) = 1;
    else
        predict(i) = 0;
    end
    if predict(i)==1 && labels(i)==0
        fp = fp + 1;
    elseif predict(i)==0 && labels(i)==1
        fn = fn + 1;
    elseif predict(i)==0 && labels(i)==0
        tn = tn + 1;
    else
        tp = tp + 1;
    end

end

accuracy = 0;
fpr = 0;
fnr = 0;

disp('Confusion Matrix')
[tn fp;fn tp]
accuracy = (tp+tn)/size(labels,1);
fpr = fp/size(labels(labels(:)==0),1);
fnr = fn/size(labels(labels(:)==1),1);
disp('Accuracy FPR FNR')
[accuracy fpr fnr]

end