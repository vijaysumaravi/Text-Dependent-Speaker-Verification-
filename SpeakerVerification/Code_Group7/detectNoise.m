function [noisy] = detectNoise(s1,s2)
%Determines whether a pair of speech signals is noisy
%
% Inputs:   s1              speech signal 1
%           s2              speech signal 2
%
% Outputs:  noisy           speech signal pair is noisy (bool)
    
    if nargin < 3 
        threshold = 1050;
    end
    if nargin < 4 
        fs = 8000;
    end
    
    s1 = audioread(char(s1));
    s2 = audioread(char(s2));
    
    ninc=round(0.016*fs);       % frame increment [fs=sample frequency]
    ovf=2;                      % overlap factor
    
    f1=rfft(enframe(s1,hanning(ovf*ninc,'periodic'),ninc),ovf*ninc,2);
    f1=f1.*conj(f1);            % convert to power spectrum
    x1=estnoiseg(f1,ninc/fs);   % Voicebox noise estimator function
    out1 = sum(sum(x1));
    
    f2=rfft(enframe(s2,hanning(ovf*ninc,'periodic'),ninc),ovf*ninc,2);
    f2=f2.*conj(f2);            % convert to power spectrum
    x2=estnoiseg(f2,ninc/fs);   % Voicebox noise estimator function
    out2 = sum(sum(x2));
    
    if (out1 > threshold || out2 > threshold)
        noisy = true;
    else
        noisy = false;
    end
end