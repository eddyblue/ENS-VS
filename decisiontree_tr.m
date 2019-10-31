function ctree = decisiontree_tr(fea_tr,lab_tr)
trnum = size(fea_tr,1);


ctree = ClassificationTree.fit(fea_tr, lab_tr);
end