tic

close;
%%clear all;


 happen=160;

 %%数据预处理
 d00=importdata('d00.dat');
 d08=importdata('d01_te.dat');
 X=d00';
 XT=d08;
[X,mean,std]=zscore(X);
XT=(XT-ones(size(XT,1),1)*mean)./(ones(size(XT,1),1)*std);
%%对测试数据 (XT) 使用训练数据的均值和标准差进行标准化。

%%主成分分析
[COEFF, SCORE, LATENT] = pca(X);
%%COEFF：主成分的系数矩阵（特征向量）。
%%SCORE：训练数据在主成分空间的投影。
%%LATENT：每个主成分的特征值，表示解释的方差。

percent = 0.85;
beta=0.99;   
k=0;
for i=1:size(LATENT,1)    
    alpha(i)=sum(LATENT(1:i))/sum(LATENT);
    if alpha(i)>=percent  
        k=i;
        break;  
    end 
end

P=COEFF(:,1:k);          
%%累积方差贡献率 (alpha) 用于选择前 k 个主成分，确保累积方差贡献率大于设定阈值（percent=0.85）。
%%选定的 P 是前 k 个主成分的系数矩阵。


%% PCA 指标计算  训练数据 T² 和 SPE（Squared Prediction Error）
for i=1:size(X,1)
    t2(i)=X(i,:)*P*inv(diag(LATENT(1:k)))*P'*X(i,:)';   
    SPE(i)=X(i,:)*COEFF*(X(i,:)*COEFF)'-X(i,:)*P*(X(i,:)*P)';
end
%%对测试数据计算 T² 和 SPE 指标，用于后续判断异常样本。
for i=1:size(XT,1);
    XTt2(i)=XT(i,:)*P*inv(diag(LATENT(1:k)))*P'*XT(i,:)';
    XTSPE(i)=XT(i,:)*COEFF*(XT(i,:)*COEFF)'-XT(i,:)*P*(XT(i,:)*P)';
end

%% compute limit of SPE T2 阈值计算

    [bandwidth,density,xmesh,cdf]=kde(t2);
    r=0.99;
    for i=1:size(cdf,1),
        if cdf(i,1)>=r,
            break;
        end;
    end;
    T2limit=xmesh(i);
    %%使用核密度估计（KDE）计算 T² 和 SPE 的分布。确定 99% 的置信限，即 T² 和 SPE 超过此值的概率为 1%。
    [bandwidth,density,xmesh,cdf]=kde(SPE);
    r=0.99;
    for i=1:size(cdf,1),
        if cdf(i,1)>=r,
            break;
        end;
    end;
    SPElimit= xmesh(i);

%%可视化 T² 和 SPE 的可视化结果展示了测试样本中异常点的位置。
figure(11)
subplot(2,1,1);
plot(1:happen,XTt2(1:happen),'b',happen+1:size(XTt2,2),XTt2(happen+1:end),'b');
hold on;
TS=T2limit*ones(size(XT,1),1);
plot(TS,'k--');
title('PCA-T2 for TE data');
xlabel('Sample');
ylabel('T2');
hold off;
subplot(2,1,2);
plot(1:happen,XTSPE(1:happen),'b',happen+1:size(XTSPE,2),XTSPE(happen+1:end),'b');
hold on;
S=SPElimit*ones(size(XT,1),1);
plot(S,'k--');
title('PCA-SPE for TE data');
xlabel('Sample');
ylabel('SPE');
hold off;

%% 异常检测错误报警率：分别统计前 happen 个样本（正常样本）中 T² 和 SPE 超出阈值的样本比例。说明模型对正常样本的误判程度。
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


%%漏报率：统计后续异常样本中 T² 和 SPE 未能检测到的样本比例。表明模型漏报异常的严重程度。
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
 disp('----PCA--False alarm rate----');
falserate_pca=[falserate_pca_T2 falserate_pca_SPE]
 disp('----PCA--Miss alarm rate----');
missrate_pca=[missrate_pca_T2 missrate_pca_SPE]
% toc
%%检测时间 检测到连续 6 个 T²/SPE 超过阈值的样本时，记录检测异常的时间点。
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
detection_time_T2;
detection_time_SPE;
runtime=toc;       