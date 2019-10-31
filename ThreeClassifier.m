function [ensembleClassifier, result_val, result_ts,F1Score] = ThreeClassifier(fea_tr, lab_tr,fea_val, lab_val,fea_ts,lab_ts)


trnum=size(fea_tr,1);
valnum=size(fea_val,1);
tsnum=size(fea_ts,1);
%初始化
m=1;
ensembleSVM.alpha = zeros(1,m);
ensembleSVM.hypo_tr=zeros(m,trnum);
ensembleSVM.I_tr=zeros(m,trnum);
ensembleSVM.hypo_val=zeros(m,valnum);
ensembleSVM.I_val=zeros(m,valnum);
ensembleSVM.hypo_ts=zeros(m,tsnum);
ensembleSVM.I_ts=zeros(m,tsnum);
ensembleSVM.thresh = 0;

ensembleCtree.alpha = zeros(1,m);
ensembleCtree.hypo_tr=zeros(m,trnum);
ensembleCtree.I_tr=zeros(m,trnum);
ensembleCtree.hypo_val=zeros(m,valnum);
ensembleCtree.I_val=zeros(m,valnum);
ensembleCtree.hypo_ts=zeros(m,tsnum);
ensembleCtree.I_ts=zeros(m,tsnum);
ensembleCtree.thresh = 0;

ensemblefisher.alpha = zeros(1,m);
ensemblefisher.hypo_tr=zeros(m,trnum);
ensemblefisher.I_tr=zeros(m,trnum);
ensemblefisher.hypo_val=zeros(m,valnum);
ensemblefisher.I_val=zeros(m,valnum);
ensemblefisher.hypo_ts=zeros(m,tsnum);
ensemblefisher.I_ts=zeros(m,tsnum);
ensemblefisher.thresh = 0;


 weightSVM =ones(trnum,1) /trnum;

 weightCtree =ones(trnum,1) /trnum;

 weightfisher =ones(trnum,1) /trnum;

          
%Traning SVM
           % resample data
           [fea_w, lab_w, idx_w] = resample(fea_tr, lab_tr, weightSVM);   
           SVMStruct = svm_tr(fea_w,lab_w);
        
           [I_tr, hypo_tr] = svm_ts(fea_tr, lab_tr, SVMStruct);
            [I_val, hypo_val] = svm_ts(fea_val, lab_val, SVMStruct);
           [I_ts, hypo_ts] = svm_ts(fea_ts, lab_ts, SVMStruct);
     
          % calculate err
          err = sum(weightSVM .* I_tr) / sum(weightSVM);
          ensembleSVM.alpha(m) = 0.5*log((1-err)/err);
         flag = (((I_tr > 0) * -2) + 1);
         weightSVM = weightSVM .* exp(-ensembleSVM.alpha(m)*flag);
         weightSVM = weightSVM ./ sum(weightSVM);  
   
ensembleSVM.hypo_tr(m,:)=hypo_tr';
ensembleSVM.I_tr(m,:)=I_tr';
ensembleSVM.hypo_val(m,:)=hypo_val';
ensembleSVM.I_val(m,:)=I_val';
ensembleSVM.hypo_ts(m,:)=hypo_ts';
ensembleSVM.I_ts(m,:)=I_ts';

     
%Train Decisiontree
          % resample data
           [fea_w, lab_w, idx_w] = resample(fea_tr, lab_tr, weightCtree);   
           CtreeStruct =decisiontree_tr(fea_w,lab_w);
 
           [I_tr, hypo_tr] = decisiontree_ts(fea_tr, lab_tr, CtreeStruct);
           [I_val, hypo_val] = decisiontree_ts(fea_val, lab_val, CtreeStruct);
           [I_ts, hypo_ts] = decisiontree_ts(fea_ts, lab_ts,CtreeStruct);
     
          % calculate err
          err = sum(weightCtree .* I_tr) / sum(weightCtree); 
          ensembleCtree.alpha(m) = 0.5*log((1-err)/err);
     
          flag = (((I_tr > 0) * -2) + 1);
          weightCtree = weightCtree .* exp(-ensembleCtree.alpha(m)*flag);
          weightCtree = weightCtree ./ sum(weightCtree);  
        
          ensembleCtree.hypo_tr(m,:)=hypo_tr';
ensembleCtree.I_tr(m,:)=I_tr';
ensembleCtree.hypo_val(m,:)=hypo_val';
ensembleCtree.I_val(m,:)=I_val';
ensembleCtree.hypo_ts(m,:)=hypo_ts';
ensembleCtree.I_ts(m,:)=I_ts';


            
%Train Fisher
          % resample data
           [fea_w, lab_w, idx_w] = resample(fea_tr, lab_tr, weightfisher);   
           [w,threshold] = fisher_tr(fea_w,lab_w);
           [I_tr, hypo_tr] = fisher_ts(fea_tr, lab_tr,w,threshold);
           [I_val, hypo_val] = fisher_ts(fea_val, lab_val,w,threshold);
           [I_ts, hypo_ts] =  fisher_ts(fea_ts, lab_ts,w,threshold);
          % calculate err
          err = sum(weightfisher .* I_tr) / sum(weightfisher);
          ensemblefisher.alpha(m) =0.5*log((1-err)/err);
         flag = (((I_tr > 0) * -2) + 1);
        weightfisher = weightfisher .* exp(-ensemblefisher.alpha(m)*flag);
        weightfisher = weightfisher ./ sum(weightfisher);  
        
ensemblefisher.hypo_tr(m,:)=hypo_tr';
ensemblefisher.I_tr(m,:)=I_tr';
ensemblefisher.hypo_val(m,:)=hypo_val';
ensemblefisher.I_val(m,:)=I_val';
ensemblefisher.hypo_ts(m,:)=hypo_ts';
ensemblefisher.I_ts(m,:)=I_ts';

 

 


%training set 分别计算每种分类器的Fscore
result_tr_SVM=ensembleSVM.alpha*ensembleSVM.hypo_tr;
result_tr_Ctree=ensembleCtree.alpha*ensembleCtree.hypo_tr;
result_tr_fisher=ensemblefisher.alpha*ensemblefisher.hypo_tr;

result_tr_SVM=result_tr_SVM';
result_tr_I_SVM=sign(result_tr_SVM); %返回训练集预测实数值；
tr_err_SVM=(result_tr_I_SVM~=lab_tr);
trerr_SVM=sum(tr_err_SVM)/trnum;
 FindPositive=2;
tr_sensitive_SVM=sum(result_tr_I_SVM+lab_tr==FindPositive)/sum(lab_tr==1);
tr_precision_SVM=sum(result_tr_I_SVM+lab_tr==FindPositive)/sum(result_tr_I_SVM==1);
if tr_precision_SVM==0 && tr_sensitive_SVM==0
 F1Score_tr_SVM=0
else
F1Score_tr_SVM= 2.*tr_precision_SVM.*tr_sensitive_SVM./(tr_precision_SVM+tr_sensitive_SVM);
end
result_tr_Ctree=result_tr_Ctree';
result_tr_I_Ctree=sign(result_tr_Ctree);
tr_err_Ctree=(result_tr_I_Ctree~=lab_tr);
trerr_Ctree=sum(tr_err_Ctree)/trnum;
 FindPositive=2;
tr_sensitive_Ctree=sum(result_tr_I_Ctree+lab_tr==FindPositive)/sum(lab_tr==1);
tr_precision_Ctree=sum(result_tr_I_Ctree+lab_tr==FindPositive)/sum(result_tr_I_Ctree==1);
if tr_precision_Ctree==0 && tr_sensitive_Ctree==0
 F1Score_tr_Ctree=0
else
F1Score_tr_Ctree= 2.*tr_precision_Ctree.*tr_sensitive_Ctree./(tr_precision_Ctree+tr_sensitive_Ctree);
end

result_tr_fisher=result_tr_fisher';
result_tr_I_fisher=sign(result_tr_fisher);
tr_err_fisher=(result_tr_I_fisher~=lab_tr);
trerr_fisher=sum(tr_err_fisher)/trnum;
 FindPositive=2;
tr_sensitive_fisher=sum(result_tr_I_fisher+lab_tr==FindPositive)/sum(lab_tr==1);
tr_precision_fisher=sum(result_tr_I_fisher+lab_tr==FindPositive)/sum(result_tr_I_fisher==1);
if tr_precision_fisher==0 && tr_sensitive_fisher==0
 F1Score_tr_fisher=0
else
F1Score_tr_fisher= 2.*tr_precision_fisher.*tr_sensitive_fisher./(tr_precision_fisher+tr_sensitive_fisher);
end
  %validating set 结果
result_val_SVM=ensembleSVM.alpha*ensembleSVM.hypo_val;
result_val_Ctree=ensembleCtree.alpha*ensembleCtree.hypo_val;
result_val_fisher=ensemblefisher.alpha*ensemblefisher.hypo_val;

result_val_SVM=result_val_SVM';
result_val_I_SVM=sign(result_val_SVM); %返回训练集预测实数值；
val_err_SVM=(result_val_I_SVM~=lab_val);
valerr_SVM=sum(val_err_SVM)/valnum;
 FindPositive=2;
val_sensitive_SVM=sum(result_val_I_SVM+lab_val==FindPositive)/sum(lab_val==1)
val_precision_SVM=sum(result_val_I_SVM+lab_val==FindPositive)/sum(result_val_I_SVM==1)
if val_precision_SVM==0 && val_sensitive_SVM==0
 F1Score_val_SVM=0
else
F1Score_val_SVM= 2.*val_precision_SVM.*val_sensitive_SVM./(val_precision_SVM+val_sensitive_SVM)
end
result_val_Ctree=result_val_Ctree';
result_val_I_Ctree=sign(result_val_Ctree);
val_err_Ctree=(result_val_I_Ctree~=lab_val);
valerr_Ctree=sum(val_err_Ctree)/valnum;
 FindPositive=2;
val_sensitive_Ctree=sum(result_val_I_Ctree+lab_val==FindPositive)/sum(lab_val==1)
val_precision_Ctree=sum(result_val_I_Ctree+lab_val==FindPositive)/sum(result_val_I_Ctree==1)
if val_precision_Ctree==0 && val_sensitive_Ctree==0
 F1Score_val_Ctree=0
else
F1Score_val_Ctree= 2.*val_precision_Ctree.*val_sensitive_Ctree./(val_precision_Ctree+val_sensitive_Ctree)
end

result_val_fisher=result_val_fisher';
result_val_I_fisher=sign(result_val_fisher);
val_err_fisher=(result_val_I_fisher~=lab_val);
valerr_fisher=sum(val_err_fisher)/valnum;
 FindPositive=2;
val_sensitive_fisher=sum(result_val_I_fisher+lab_val==FindPositive)/sum(lab_val==1);
val_precision_fisher=sum(result_val_I_fisher+lab_val==FindPositive)/sum(result_val_I_fisher==1);
if val_precision_fisher==0 && val_sensitive_fisher==0
 F1Score_val_fisher=0
else
F1Score_val_fisher= 2.*val_precision_fisher.*val_sensitive_fisher./(val_precision_fisher+val_sensitive_fisher);
end  
%testing set 结果 
result_ts_SVM=ensembleSVM.alpha*ensembleSVM.hypo_ts;
result_ts_Ctree=ensembleCtree.alpha*ensembleCtree.hypo_ts;
result_ts_fisher=ensemblefisher.alpha*ensemblefisher.hypo_ts;

result_ts_SVM=result_ts_SVM';
result_ts_I_SVM=sign(result_ts_SVM); %返回训练集预测实数值；
ts_err_SVM=(result_ts_I_SVM~=lab_ts);
tserr_SVM=sum(ts_err_SVM)/tsnum;
 FindPositive=2;
ts_sensitive_SVM=sum(result_ts_I_SVM+lab_ts==FindPositive)/sum(lab_ts==1);
ts_precision_SVM=sum(result_ts_I_SVM+lab_ts==FindPositive)/sum(result_ts_I_SVM==1);
if ts_precision_SVM==0 && ts_sensitive_SVM==0
 F1Score_ts_SVM=0
else
F1Score_ts_SVM= 2.*ts_precision_SVM.*ts_sensitive_SVM./(ts_precision_SVM+ts_sensitive_SVM);
end
result_ts_Ctree=result_ts_Ctree';
result_ts_I_Ctree=sign(result_ts_Ctree);
ts_err_Ctree=(result_ts_I_Ctree~=lab_ts);
tserr_Ctree=sum(ts_err_Ctree)/tsnum;
 FindPositive=2;
ts_sensitive_Ctree=sum(result_ts_I_Ctree+lab_ts==FindPositive)/sum(lab_ts==1);
ts_precision_Ctree=sum(result_ts_I_Ctree+lab_ts==FindPositive)/sum(result_ts_I_Ctree==1);
if ts_precision_Ctree==0 && ts_sensitive_Ctree==0
 F1Score_ts_Ctree=0
else
F1Score_ts_Ctree= 2.*ts_precision_Ctree.*ts_sensitive_Ctree./(ts_precision_Ctree+ts_sensitive_Ctree);
end

result_ts_fisher=result_ts_fisher';
result_ts_I_fisher=sign(result_ts_fisher);
ts_err_fisher=(result_ts_I_fisher~=lab_ts);
tserr_fisher=sum(ts_err_fisher)/tsnum;
 FindPositive=2;
ts_sensitive_fisher=sum(result_ts_I_fisher+lab_ts==FindPositive)/sum(lab_ts==1);
ts_precision_fisher=sum(result_ts_I_fisher+lab_ts==FindPositive)/sum(result_ts_I_fisher==1);
if ts_precision_fisher==0 && ts_sensitive_fisher==0
 F1Score_ts_fisher=0
else
F1Score_ts_fisher= 2.*ts_precision_fisher.*ts_sensitive_fisher./(ts_precision_fisher+ts_sensitive_fisher);
end

if F1Score_val_SVM>=F1Score_val_Ctree &&F1Score_val_SVM>=F1Score_val_fisher
 result_val=result_val_I_SVM;
 result_ts=result_ts_I_SVM;
 ensembleClassifier=ensembleSVM;
 F1Score=F1Score_val_SVM;

elseif F1Score_val_Ctree>=F1Score_val_SVM &&F1Score_val_SVM>=F1Score_val_fisher

      result_val=result_val_Ctree;
      result_ts=result_ts_Ctree;
      ensembleClassifier=ensembleCtree;
      F1Score=F1Score_val_Ctree;
elseif F1Score_val_fisher>=F1Score_val_Ctree &&F1Score_val_fisher>=F1Score_val_SVM
                 result_val=result_val_fisher;
                 result_ts=result_ts_fisher;
                ensembleClassifier=ensemblefisher;
                F1Score=F1Score_val_fisher;
else  
      result_val=result_val_Ctree;
      result_ts=result_ts_Ctree;
      ensembleClassifier=ensembleCtree;
      F1Score=0;
end

