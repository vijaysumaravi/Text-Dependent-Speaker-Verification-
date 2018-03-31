function featureDict = extractFeatures(fileList,v_thres,features_opt,voiced_only)
%Extract pitch, formants
%
% Inputs:   fileList        input file list
%           v_thres         threshold for voiced frame
%           features_opt    optional features requested (array of binary)
%                           [lpc plp gfcc]
%           voiced_only     return features for voiced frames only (bool)
%
% Outputs:  featureDict     features returned as dictionary

featureDict = containers.Map;

for i = 1:length(fileList)
    
    % Read input file
    [s,fs] = audioread(fileList{i});
    
    % Extract pitch
    [F0,~] = fast_mbsc_fixedWinlen_tracking(s,fs,1,v_thres);

    % Extract formants (autocorr)
    [formants,lpc] = extractFormants(s,fs);

    % Extract MFCCs
    [mfcc] = melcepst(s,fs,'EdD');

    numberOfFrames = min([size(F0,1),size(formants,1),size(mfcc,1)]);

    % Pitch, formants, mfcc
    features = horzcat(F0(1:numberOfFrames),formants(1:numberOfFrames,:),...
        mfcc(1:numberOfFrames,:));

    % If LPC requested
    if features_opt(1) == 1
        features = horzcat(features,lpc(1:numberOfFrames,:));
    end
    
    % If PLP requested
    if features_opt(2) == 1
        plp = extractPLP(s,fs,10)';
        numberOfFrames = min(numberOfFrames,size(plp,1));
        features = horzcat(features(1:numberOfFrames,:),plp(1:numberOfFrames,:));
    end
    
    % If GFCC requested
    if features_opt(3) == 1
        gfcc = gtfeatures(s,fs)';
        numberOfFrames = min(numberOfFrames,size(gfcc,1));
        features = horzcat(features(1:numberOfFrames,:), gfcc(1:numberOfFrames,:));
    end

    % If only voiced frames required
    if voiced_only
        features = features(features(:,1)~=0,:);
    end
        
    featureDict(fileList{i}) = features;
    
    if(mod(i,10)==0)
        disp(['Completed ',num2str(i),' of ',num2str(length(fileList)),' files.']);
    end
end

end