function AUC = CalculateAUC(rates)
% To calculate AUC values
% Input:
%   rates: (fpr,tpr) vector with fpr in ascending order
% Output:
%   AUC: AUC value

% Copyright: Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)

AUC = 0;
for i=1:size(rates,1)-1
    AUC = AUC + (rates(i+1,1)-rates(i,1))*(rates(i+1,2)+rates(i,2))/2;
end