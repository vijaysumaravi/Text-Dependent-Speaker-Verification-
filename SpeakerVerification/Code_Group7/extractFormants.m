function [formants,lpcc] = extractFormants(s,fs)
%Extract formant frequencies
%
% Inputs:   s           speech signal
%           fs          sampling frequency in Hz
%
% Outputs:  formants    formant frequencies

% Order 12, frame size 160, frame rate 80
[ar,~,~] = lpcauto(s,10,[80,160]);

formants = zeros(size(ar,1),4);

for i=1:size(ar,1)
    formants(i,:) = formantsLPC(ar(i,:),fs);
end

lpcc = ar(:,2:end);

end