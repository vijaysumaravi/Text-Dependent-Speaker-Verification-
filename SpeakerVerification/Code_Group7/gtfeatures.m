function[gfcc] =  gtfeatures(snd, sampFreq, numChannel)
% Generate gammatone features (GF) and gammatone frequency cepstral
% coefficients (GFCC).

% ctl_list: list of files to be processed
% sampFreq: sampling frequency, default is 8000
% bWav: boolean variable indicating whether speech data is stored in the
%       WAV format
% numChannel: number of channels

% Written by Yang Shao, and adapted by Xiaojia Zhao in Oct'11


if ~exist('sampFreq', 'var')
    sampFreq = 8000;
end

if ~exist('numChannel', 'var')
    numChannel = 64;
end

gt = gen_gammaton(sampFreq, numChannel);  % get gammatone filterbank

sig = reshape(snd, 1, length(snd));  % gammatone filtering function requires input to be row vector

g=fgammaton(sig, gt, sampFreq, numChannel);     % gammatone filter pass and decimation to get GF features

gfcc = gtf2gtfcc(g, 2, 23);  % apply dct to get GFCC features with 0th coefficient removed
       
end
