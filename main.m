
clc;clear
warning off
FindPositive=2;
FindNegtive=-2;
 auc_ts=[];
 auc_val=[];
 Num_iter=1; %十折交叉验证
  T_const=300;   %64
  T=T_const;

  pos_iter=1;

load data.mat

rand('state',sum(100*clock));    %按系统时间设置随机数种子 

 auc_val=[];
 auc=[];
lab_all_val{pos_iter,1}=lab_val;
lab_all_ts{pos_iter,1}=lab_ts;


poscount = sum(lab_tr==1);
negcount = length(lab_tr)-poscount;
posset = fea_tr(lab_tr==1,:);
negset = fea_tr(lab_tr==-1,:);

% k-means     %%%%%%%%%%%%%%%%%%%%%%%
    X=fea_tr;
    % set algorithm parameters
    TOL = 0.0004;
    ITER = 30;
    kappa =30;      
    % run k-Means on random data
    tic;
    [C, I, iter] = myKmeans(X, kappa, ITER, TOL);
    toc    
    % show number of iteration taken by k-means
    disp(['k-means instance took ' int2str(iter) ' iterations to complete']);        
    std_tmp=[];
for k=1:kappa 
    std=std2(X(find(I == k), :));
    std_tmp=[std_tmp std];
end

count_tmp=[];
for k=1:kappa
    count=sum(I==k);
    count_tmp=[count_tmp count];
end

num_tmp=[];
for k=1:kappa
    num=round(poscount*std_tmp(k)*count_tmp(k)/(std_tmp*count_tmp'));
    num_tmp=[num_tmp num];
end


[result_val result_ts]=Ensemble(X,I,kappa,num_tmp,posset,fea_val,lab_val,fea_ts,lab_ts,T);
%f = EvaluateValue(fea_ts,lab_ts,ensemble); % get real valued output
prob_all_val{pos_iter,1}=result_val';
prob_all_ts{pos_iter,1}=result_ts';
%validation set结果
rates_val = CalculatePositives(lab_val,result_val);
%plot(rates(:,1),rates(:,2));
auc_val = [auc_val,CalculateAUC(rates_val)];
hypo=sign(result_val);
TP_val(pos_iter)= sum((hypo+lab_val)==FindPositive);
TN_val(pos_iter)= sum((hypo+lab_val)==FindNegtive);
P_val(pos_iter)=sum(lab_val==1);
N_val(pos_iter)=sum(lab_val==-1);
FP_val(pos_iter)=N_val(pos_iter)-TN_val(pos_iter);
FN_val(pos_iter)=P_val(pos_iter)-TP_val(pos_iter);
%test set结果
rates = CalculatePositives(lab_ts,result_ts);
%plot(rates(:,1),rates(:,2));
auc = [auc,CalculateAUC(rates)];
hypo=sign(result_ts);
TP(pos_iter)= sum((hypo+lab_ts)==FindPositive);
TN(pos_iter)= sum((hypo+lab_ts)==FindNegtive);
P(pos_iter)=sum(lab_ts==1);
N(pos_iter)=sum(lab_ts==-1);
FP(pos_iter)=N(pos_iter)-TN(pos_iter);
FN(pos_iter)=P(pos_iter)-TP(pos_iter);


 %计算评价指标
accuracy_val=(TP_val+TN_val)/length(lab_val);
sensitive_val=TP_val./P_val;
precision_val=TP_val./(TP_val+FP_val);
specificity_val=TN_val./N_val;
F1Score_val = 2.*precision_val.*sensitive_val./(precision_val+sensitive_val);
accuracy_mean_val=mean(accuracy_val);
sensitive_mean_val=mean(sensitive_val);
precision_mean_val=mean(precision_val);
specificity_mean_val=mean(specificity_val);
F1Score_mean_val=mean(F1Score_val);
auc_mean_val=mean(auc_val);
evaluate_val=[accuracy_mean_val sensitive_mean_val precision_mean_val specificity_mean_val F1Score_mean_val auc_mean_val];

accuracy=(TP+TN)./length(lab_ts);
sensitive=TP./P;
precision=TP./(TP+FP);
specificity=TN./N;
F1Score = 2.*precision.*sensitive./(precision+sensitive);
accuracy_mean=mean(accuracy);
sensitive_mean=mean(sensitive);
precision_mean=mean(precision);
specificity_mean=mean(specificity);
F1Score_mean=mean(F1Score);
auc_mean=mean(auc);
evaluate=[accuracy_mean sensitive_mean precision_mean specificity_mean F1Score_mean auc_mean];

allResult=[evaluate_val evaluate]
[num, text, raw] = xlsread('ens_evaluate.xls');
[rowN, columnN]=size(raw);
sheet=1;
xlsRange=['A',num2str(rowN+1)];
xlswrite('ens_evaluate.xls',allResult,sheet,xlsRange);
