function [w, threshold] = fisher_tr(fea_tr,lab_tr)
trnum = size(fea_tr,1);
if (trnum ~= size(lab_tr, 1))
    disp('training set has different feature and label size\n');
    return;
end

fea_tr1_no = 0;
fea_tr2_no = 0;
for i = 1:trnum,
    if lab_tr(i) == min(lab_tr)
        fea_tr1_no = fea_tr1_no + 1;
        fea_tr1(fea_tr1_no,:) = fea_tr(i,:);
    else
        fea_tr2_no = fea_tr2_no + 1;
        fea_tr2(fea_tr2_no,:) = fea_tr(i,:);
    end
end

m = mean(fea_tr1);
m(2:2,:) = mean(fea_tr2);

fea_tr1_no = size(fea_tr1,1);
fea_tr2_no = size(fea_tr2,1);
feature_dim = size(fea_tr1,2);  

ssw = zeros(feature_dim,feature_dim);
for i = 1:fea_tr1_no,
    ssw = ssw + (fea_tr1(i:i,:)-m(1:1,:))'*(fea_tr1(i:i,:)-m(1:1,:));
end
for i = 1:fea_tr2_no,
    ssw = ssw + (fea_tr2(i:i,:)-m(2:2,:))'*(fea_tr2(i:i,:)-m(2:2,:));
end

w = pinv(ssw) * (m(1:1,:) - m(2:2,:))';
result_train = fea_tr* w;

theta = w' * (m(1:1,:) + m(2:2,:))' / 2;
theta(:,2:2) = w' * (fea_tr1_no * m(1:1,:) + fea_tr2_no * m(2:2,:))' / trnum;
theta(:,3:3) = w' * (fea_tr2_no * m(1:1,:) + fea_tr1_no * m(2:2,:))' / trnum;
xigma = std(result_train(1:fea_tr1_no,1:1));
xigma(2) = std(result_train(fea_tr1_no + 1:trnum,1:1));
theta(:,4:4) = w' * (xigma(1) * m(1:1,:) + xigma(2) * m(2:2,:))' / (xigma(1) + xigma(2));
theta(:,5:5) = w' * (xigma(1) * m(2:2,:) + xigma(2) * m(1:1,:))' / (xigma(1) + xigma(2));
threshold = theta(:,1:1);
end