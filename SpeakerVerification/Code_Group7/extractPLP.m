function [lpcas, spectra, pspectrum, F, M] = plp_extraction(samples, sr, modelorder)

%%
%lpacas = lpcs of audio signal (not lpcc)
%spectra can be plotted using imagesc, compares to signal spectogram 

%modelorder = order of the lpc coeff of the perceived signal

%%


if nargin < 2
  sr = 8000;
end

if nargin < 3
  modelorder = 10;
end

% first compute power spectrum
pspectrum = powspec(samples, sr);

% next group to critical bands
aspectrum = audspec(pspectrum, sr);
nbands = size(aspectrum,1);
  
% do final auditory compressions
postspectrum = postaud(aspectrum, sr/2); % 2012-09-03 bug: was sr

if modelorder > 0

  % LPC analysis of perceived signal
  lpcas = dolpc(postspectrum, modelorder);

  % .. or to spectra
  [spectra,F,M] = lpc2spec(lpcas, nbands);

end
