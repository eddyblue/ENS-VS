% function resample 
function [fea_w, lab_w, idx_w] = resample(fea, lab, weight)
% resample feature set based on the distribution indicated by weight
% the weights should be sum to one
% fea:      vsize*trainnum
% lab:      1    *trainnum
% weight:   1    *trainnum
% return resampled feature, label, and index

% first check the size of fea and weight
fsize = size(fea,1);
if (fsize ~= size(weight, 1))
    disp('weight size does not equal to feature size\n');
    return;
end

% first check the sum of the weight
s = sum(weight);
if ( abs(s-1) > 1e-5 )
    disp('weight sum is not equral to 1, normalizing...\n');
    weight = weight ./ s;
end

% resample 
fea_w = []; lab_w = []; idx_w = zeros(fsize, 1);

wheel = zeros(fsize, 1);        % accumulative weights
wheel(1) = weight(1);
for i = 2:fsize
    wheel(i) = wheel(i-1) + weight(i);
end

for t = 1:fsize
    r = rand;       % pick 
    bot = 1; top = fsize;
    while (wheel(bot)<r && top>=bot)     % binary search
        i = floor((bot+top)/2);
        if (r < wheel(i))
            top = i - 1;
        else
            bot = i + 1;
        end
    end
    fea_w = [fea_w; fea(bot, :)];
    lab_w = [lab_w; lab(bot, 1)];
    idx_w(t) = bot;
end