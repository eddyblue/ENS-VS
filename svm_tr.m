function SVMStruct = svm_tr(fea_tr,lab_tr)
trnum = size(fea_tr,1);

%参数设置很重要，默认参数效果很差。
%SVMStruct = svmtrain(fea_tr, lab_tr,'kernel_function','rbf','rbf_sigma',0.5);

SVMStruct = svmtrain( fea_tr,lab_tr);



end