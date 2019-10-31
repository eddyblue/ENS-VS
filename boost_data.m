function [boostset, boosttarget] = boost_data(trainset, traintarget,weight)
% To sample a subset according to each example's weight
% Input:
%   trainset: n-by-d training set
%   traintarget: n-by-1 training target
%   weight: n-by-1 weight vector, with its sum is 1
% Output:
%   boostset: sampled data set
%   boosttarget: labels for boostset



n = length(traintarget);
c_sum = cumsum(weight);  %�����ۼӵĺ�

select = rand(size(traintarget));   %����һ����traintarget�ߴ�һ�����������
for i=1:n
    select(i) = min(find(c_sum>=select(i)));    
end

boostset = trainset(select,:);
boosttarget = traintarget(select);
