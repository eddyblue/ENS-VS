function [I, hypo] = fisher_ts(fea_ts, lab_ts, w, threshold)
tsnum = size(fea_ts, 1);
if (tsnum ~= size(lab_ts, 1))
    disp('testing set has different feature and label size\n');
    return;
end

result_test = fea_ts * w;
hypo = -sign(result_test - threshold);
I = (hypo ~= lab_ts);
end