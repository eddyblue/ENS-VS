function value = EvaluateValue(dataset,datasettarget,ensemble)
% To predict real values for a set of examples using 
% AdaBoost/EasyEnsemble/BalanceCascade classifer
% Input:
%   dataset: n-by-d test set
%   ensemble: AdaBoost/EasyEnsemble/BalanceCascade classifer
% Output:
%   values: predicted real values of test examples

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)

value = zeros(size(dataset,1),1);
for i=1:length(ensemble.trees)
    [I,testResult]=svm_ts(dataset, datasettarget, ensemble.trees{i});
    value = value + ensemble.alpha(i) *testResult;
end