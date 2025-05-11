tic

close;
clear all;

 happen=160;
 d00=importdata('d00.dat');
 d08=importdata('d01_te.dat');
 X=d00';
 XT=d08;
[X,mean1,std1]=zscore(X);
XT=(XT-ones(size(XT,1),1)*mean1)./(ones(size(XT,1),1)*std1);


%% Kernel PCA
percent = 0.85;
options.KernelType = 'Gaussian';
for i = 1 : size(X,1)
    for j = 1 : size(X,1)
        dist(i,j) = norm(X(i,:)-X(j,:))^2;
    end
end
options.t = 10*52*mean(std1);
options.ReducedDim = 500;
[EVC,LATENT] = KPCA(X,options);
Ktest = constructKernel(X,X,options);
TKtest = constructKernel(XT,X,options);

k=0;
for i=1:size(LATENT,1)     
    alpha(i)=sum(LATENT(1:i))/sum(LATENT);
    if alpha(i)>=percent  
        k=i;
        break;  
    end 
end
P = EVC(:,1:k);
for i=1:size(Ktest,1)
    t2(i)=Ktest(i,:)*P*inv(diag(LATENT(1:k)))*P'*Ktest(i,:)';  
    SPE(i)=Ktest(i,:)*Ktest(i,:)'-Ktest(i,:)*P*(Ktest(i,:)*P)';
    SPE(i)= norm(Ktest(i,:)-Ktest(i,:)*P*P')^2; 
end

for i=1:size(TKtest,1)
    XTt2(i)=TKtest(i,:)*P*inv(diag(LATENT(1:k)))*P'*TKtest(i,:)'; 
    XTSPE(i)=TKtest(i,:)*TKtest(i,:)'-TKtest(i,:)*P*(TKtest(i,:)*P)';
    XTSPE(i)= norm(TKtest(i,:)-TKtest(i,:)*P*P')^2;
end


%% compute limit of SPE T2

    [bandwidth,density,xmesh,cdf]=kde(t2);
    r=0.99;
    for i=1:size(cdf,1),
        if cdf(i,1)>=r,
            break;
        end;
    end;
    T2limit=xmesh(i);
    
    [bandwidth,density,xmesh,cdf]=kde(SPE);
    r=0.99;
    for i=1:size(cdf,1),
        if cdf(i,1)>=r,
            break;
        end;
    end;
    SPElimit= xmesh(i);


figure(11)
subplot(2,1,1);
plot(1:happen,XTt2(1:happen),'b',happen+1:size(XTt2,2),XTt2(happen+1:end),'b');
hold on;
TS=T2limit*ones(size(XT,1),1);
plot(TS,'k--');
title('KPCA-T2 for TE data');
xlabel('Sample');
ylabel('T2');
hold off;
subplot(2,1,2);
plot(1:happen,XTSPE(1:happen),'b',happen+1:size(XTSPE,2),XTSPE(happen+1:end),'b');
hold on;
S=SPElimit*ones(size(XT,1),1);
plot(S,'k--');
title('KPCA-SPE for TE data');
xlabel('Sample');
ylabel('SPE');
hold off;
%False alarm rate
falseT2=0;
falseSPE=0;
for wi=1:happen
    if XTt2(wi)>T2limit
        falseT2=falseT2+1;
    end
    falserate_pca_T2=100*falseT2/happen;
    if XTSPE(wi)>SPElimit
        falseSPE=falseSPE+1;
    end
    falserate_pca_SPE=100*falseSPE/happen;
end
%Miss alarm rate
missT2=0;
missSPE=0;
for wi=happen+1:size(XTt2,2)
    if XTt2(wi)<T2limit
        missT2=missT2+1;
    end
    if XTSPE(wi)<SPElimit
        missSPE=missSPE+1;
    end 
end
missrate_pca_T2=100*missT2/(size(XTt2,2)-happen);
missrate_pca_SPE=100*missSPE/(size(XTt2,2)-happen);
 disp('----KPCA--False alarm rate----');
falserate_pca=[falserate_pca_T2 falserate_pca_SPE]
 disp('----KPCA--Miss alarm rate----');
missrate_pca=[missrate_pca_T2 missrate_pca_SPE]
% toc
i1=happen+1;

while i1<=size(X,1)
   T2_mw(i1,:)=XTt2(1,i1:(i1+5))-T2limit*ones(1,6);
   flag1=0;
   for j1=1:6
       if T2_mw(i1,j1)<0
           flag1=1;
           i1=i1+j1;
           break;
       end
   end
   if flag1==0
       detection_time_T2=i1-happen;
       break;
   end
end
i2=happen+1;
while i2<=size(X,1)
    SPE_mw(i2,:)= XTSPE(1,i2:(i2+5))-SPElimit*ones(1,6);
    flag2=0;
    for j2=1:6
       if SPE_mw(i2,j2)<0
           flag2=1;
           i2=i2+j2;
           break;
       end
   end
   if flag2==0
       detection_time_SPE=i2-happen;
       break;
   end
end
detection_time_T2
detection_time_SPE
runtime=toc       