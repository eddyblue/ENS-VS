function SVMStruct = svm_tr(fea_tr,lab_tr)
trnum = size(fea_tr,1);

%�������ú���Ҫ��Ĭ�ϲ���Ч���ܲ
%SVMStruct = svmtrain(fea_tr, lab_tr,'kernel_function','rbf','rbf_sigma',0.5);

SVMStruct = svmtrain( fea_tr,lab_tr);



end