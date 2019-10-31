function [I, hypo] = svm_ts(fea_ts, lab_ts, SVMStruct)
tsnum = size(fea_ts, 1);

hypo = svmclassify(SVMStruct,fea_ts);
I = (hypo ~= lab_ts);
end