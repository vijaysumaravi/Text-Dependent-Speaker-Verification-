function [formants] = formantsLPC(ar,fs)
%Extract formant frequencies from LPC coefficients
%
% Inputs:   ar          LPC coefficients
%           fs          sampling frequency in Hz
%
% Outputs:  formants    formant frequencies

r = roots(ar(1,:)); % Find roots of polynomial with ar as coefficients
r = r(imag(r)>=0); % Limit to roots corresponding to 0 Hz to fs/2
[ffreq, indices] = sort(atan2(imag(r),real(r)).*(fs/(2*pi)));
bw = -1/2*(fs/(2*pi))*log(abs(r(indices)));
nn = 1;
for kk = 1:length(ffreq)
    if (ffreq(kk) > 90 && bw(kk) <400)
        formants(nn) = ffreq(kk);
        nn = nn+1;
    end
end

% Extract only first 4 formant frequencies
n = 4;
for j=1:n
    if(length(formants)<=j)
        t = j; break;
    else
        t = n; continue;
    end
end

formants = [formants(1:t) zeros(1,n-t)];

end