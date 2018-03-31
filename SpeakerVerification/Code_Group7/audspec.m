function [aspectrum,wts] = audspec(pspectrum, sr, nfilts, fbtype, minfreq, maxfreq, sumpower, bwidth)
%[aspectrum,wts] = audspec(pspectrum, sr, nfilts, fbtype, minfreq, maxfreq, sumpower, bwidth)
%
% perform critical band analysis (see PLP)
% takes power spectrogram as input

if nargin < 2;  sr = 16000;                          end
if nargin < 3;  nfilts = ceil(hz2bark(sr/2))+1;      end
if nargin < 4;  fbtype = 'bark';  end
if nargin < 5;  minfreq = 0;    end
if nargin < 6;  maxfreq = sr/2; end
if nargin < 7;  sumpower = 1;   end
if nargin < 8;  bwidth = 1.0;   end

[nfreqs,nframes] = size(pspectrum);

nfft = (nfreqs-1)*2;

if strcmp(fbtype, 'bark')
  wts = fft2barkmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq);
end

wts = wts(:, 1:nfreqs);

% Integrate FFT bins into Mel bins, in abs or abs^2 domains:
if (sumpower)
  aspectrum = wts * pspectrum;
else
  aspectrum = (wts * sqrt(pspectrum)).^2;
end