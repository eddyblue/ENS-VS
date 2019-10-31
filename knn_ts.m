function [I, hypo] = knn_ts(fea_tr,lab_tr, fea_ts, lab_ts)
trnum = size(fea_tr, 1);
tsnum = size(fea_ts, 1);



% hypo = knnclassify(fea_ts, fea_tr, lab_tr);
hypo = knnclassify(fea_ts, fea_tr, lab_tr,1);
I = (hypo ~= lab_ts);
end