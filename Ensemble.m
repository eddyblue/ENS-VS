function [result_val, result_ts]= Ensemble (X, I,kappa,num_tmp,posset,fea_val,lab_val,fea_ts,lab_ts,T)


    resultMatrix_val=[];
    resultMatrix_ts=[];
     Fscore_val=[];
result_val= zeros(size(lab_val));
result_ts= zeros(size(lab_ts));
for node=1:T% stopping criteria
    node
    
negset=[];    
for k=1:kappa
    tmpX=X(find(I==k),:);
    tmp_X=tmpX(randperm(size(tmpX,1)),:);
     negset = [negset;tmp_X(1:num_tmp(k),:)];   
end   
curtrainset=[negset;posset];
curtarget=[-ones(size(negset,1),1);ones(size(posset,1),1)];
 
    
    [ensembleClassifier,result_val_node,result_ts_node,F1Score] = ThreeClassifier(curtrainset,curtarget,fea_val,lab_val,fea_ts,lab_ts);       
    resultMatrix_val=[resultMatrix_val result_val_node];
    resultMatrix_ts=[resultMatrix_ts result_ts_node];
    Fscore_val=[Fscore_val F1Score];
    


end

Fscore_mean=mean(Fscore_val);

[r,p] = corrcoef(resultMatrix_val);
r(find(isnan(r)==1)) = 1;
r=sum(r);
r_mean=mean(r);
index1=find(r<r_mean);
index2=find(Fscore_val>Fscore_mean);
index=intersect(index1,index2);
result_val=sum(resultMatrix_val(:,index1),2);
result_ts=sum(resultMatrix_ts(:,index1),2);

