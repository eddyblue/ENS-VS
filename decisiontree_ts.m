function [I, hypo] = decisiontree_ts(fea_ts, lab_ts, ctree)
tsnum = size(fea_ts, 1);

hypo = predict(ctree,fea_ts);
I = (hypo ~= lab_ts);
end