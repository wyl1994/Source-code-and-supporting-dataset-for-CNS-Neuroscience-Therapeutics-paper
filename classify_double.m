clc;
clear;
close all;
%% Load dataset
load('all_FE.mat');
feature = all_FE(:,1:28);
label = all_FE(:,29);

%% Feature selection
n = 10;                             % n = the number of selected features
[r,p] = corr(feature,label);
[p_sort, idx] = sort(p,1); 
% select the top n features with the highest correlation coefficients
idx_select = idx(1:n);
% idx_select = [14,10,18,16,2,22,6,26,13,27];
% These are the most frequently used feature indexes when selected for
% features from different patients.
feature_select = feature(:,idx_select);


%% K-fold cross-validation model training
k = 5;                              % k = the number of folds
num = size(feature_select,1);
u = rem(num,k);

% SVM
train_num = 0;
pre_train_SVM = [];
pre_SVM = [];
y_train = [];
y_test = [];
for i = 1:k
    if i <= u
        tmp(i) = ceil(num/k);
    else
        tmp(i) = floor(num/k);
    end
    test_data = feature_select(train_num + 1:train_num + tmp(i),:);
    train_data = [feature_select(1:train_num,:);...
        feature_select(train_num + tmp(i) + 1:num,:)];
    label_test = label(train_num + 1:train_num + tmp(i));
    y_test = [y_test;label_test];
    label_train = [label(1:train_num);label(train_num + tmp(i) + 1:num)];
    y_train = [y_train;label_train];
    SVMMODEL = fitcsvm(train_data,label_train,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Optimizer','bayesopt'),'KernelFunction','gaussian');
    pre_train_SVM = [pre_train_SVM;predict(SVMMODEL,train_data)];
    pre = predict(SVMMODEL,test_data);
    pre_SVM = [pre_SVM;pre];
    correct_num_SVM(i) = sum((pre-label_test)==0);
    accuracy_svm(i) = correct_num_SVM(i) / length(label_test);
    train_num = train_num + tmp(i);
end
for i = 1:k
    fprintf("SVM Fold No.%d, Correct number: %d, total: %d.\n", i, correct_num_SVM(i), tmp(i));
end
fprintf("ACC mean: %.4f, std: %.4f\n",mean(accuracy_svm),std(accuracy_svm));
[CM_SVM_train,~] = confusionmat(y_train, pre_train_SVM);
fprintf("svm train:\n");
[acc_train_SVM, prec_train_SVM,sen_train_SVM,spe_train_SVM,f1_train_SVM]...
    = print_performance(CM_SVM_train);
[CM_SVM_test,~] = confusionmat(y_test, pre_SVM);
fprintf("svm validate:\n");
[acc_test_SVM, prec_test_SVM,sen_test_SVM,spe_test_SVM,f1_test_SVM]...
    = print_performance(CM_SVM_test);

% Random Forest
train_num = 0;
pre_train_RF = [];
pre_RF = [];
y_train = [];
y_test = [];
for i = 1:k
    if i <= u
        tmp(i) = ceil(num/k);
    else
        tmp(i) = floor(num/k);
    end
    test_data = feature_select(train_num + 1:train_num + tmp(i),:);
    train_data = [feature_select(1:train_num,:);...
        feature_select(train_num + tmp(i) + 1:num,:)];
    label_test = label(train_num + 1:train_num + tmp(i));
    y_test = [y_test;label_test];
    label_train = [label(1:train_num);label(train_num + tmp(i) + 1:num)];
    y_train = [y_train;label_train];
    nTree = 10;
    B = TreeBagger(nTree,train_data,label_train,'OOBPredictorImportance','on',...
        'Method', 'classification', 'OOBPrediction','on', 'minleaf', 2);
    predict_train_label = predict(B,train_data);
    predict_train_label = str2double(predict_train_label);
    pre_train_RF = [pre_train_RF;predict_train_label];
    predict_label = predict(B,test_data);
    predict_label = str2double(predict_label);
    pre_RF = [pre_RF;predict_label];
    correct_num_RF(i) = length(find(predict_label == label_test));
    accuracy_RF(i) = correct_num_RF(i)/length(label_test);
    fprintf("RF Fold No.%d, Correct number: %d, total: %d.\n", i, correct_num_RF(i), tmp(i));
    train_num = train_num + tmp(i);
end
fprintf("ACC mean: %.4f, std: %.4f\n",mean(accuracy_RF),std(accuracy_RF));
[CM_RF_train,~] = confusionmat(y_train, pre_train_RF);
fprintf("RF train:\n");
[acc_train_RF, prec_train_RF,sen_train_RF,spe_train_RF,f1_train_RF]...
    = print_performance(CM_RF_train);
[CM_RF_test,~] = confusionmat(y_test, pre_RF);
fprintf("RF validate:\n");
[acc_test_RF, prec_test_RF,sen_test_RF,spe_test_RF,f1_test_RF]...
    = print_performance(CM_RF_test);

% KNN
correct_num_kNN = [];
prediction = [];
train_num = 0;
for i = 1:k
    if i <= u
        tmp(i) = ceil(num/k);
    else
        tmp(i) = floor(num/k);
    end
    test_data = feature_select(train_num + 1:train_num + tmp(i),:);
    train_data = [feature_select(1:train_num,:);...
        feature_select(train_num + tmp(i) + 1:num,:)];
    label_test = label(train_num + 1:train_num + tmp(i));
    label_train = [label(1:train_num);label(train_num + tmp(i) + 1:num)];
    mdl = KDTreeSearcher(train_data);
    [index,~] = knnsearch(mdl,test_data,'k',5);
    resultClass = [];
    for j = 1:size(index,1) 
        tempClass = label_train(index(j,:));
        result = mode(tempClass);
        resultClass(j,1) = result;
    end
    prediction = [prediction;resultClass];
    correct_num_kNN(i) = sum( label_test == resultClass );
    accuracy_kNN(i) = correct_num_kNN(i)./ size(resultClass,1);
    fprintf("KNN Fold No.%d, Correct number: %d, total: %d.\n", i, correct_num_kNN(i), tmp(i));
    train_num = train_num + tmp(i);
end
fprintf("ACC mean: %.4f, std: %.4f\n",mean(accuracy_kNN),std(accuracy_kNN));
[CM_kNN,~] = confusionmat(label, prediction);
[avr_acc_kNN,precision_kNN,sensitivity_kNN,specificity_kNN,F1_score_kNN]...
    = print_performance(CM_kNN);


% LDA
correct_num_LDA = [];
train_num = 0;
pre_train_LDA = [];
pre_LDA = [];
y_train = [];
y_test = [];
for i = 1:k
    if i <= u
        tmp(i) = ceil(num/k);
    else
        tmp(i) = floor(num/k);
    end
    test_data = feature_select(train_num + 1:train_num + tmp(i),:);
    train_data = [feature_select(1:train_num,:);...
        feature_select(train_num + tmp(i) + 1:num,:)];
    label_test = label(train_num + 1:train_num + tmp(i));
    y_test = [y_test;label_test];
    label_train = [label(1:train_num);label(train_num + tmp(i) + 1:num)];
    y_train = [y_train;label_train];
    [mappedX,mapping]=lda(train_data,label_train,1);

    figure()
    % success
    index1=find(label_train==1);
    plot(mappedX(index1,1),'or','MarkerSize',7); 
    hold on;
    % failure
    index2=find(label_train==2);
    plot(mappedX(index2,1),'^g','MarkerSize',7); 
    legend('success group','failure group');

    predict_train_LDA = zeros(size(label_train));
    predict_train_LDA(find(mappedX > 0)) = 1;
    predict_train_LDA(find(mappedX <= 0)) = 2;
    correct_num_LDA(i) = sum(predict_train_LDA == label_train);
    predict_train_LDA_tmp = zeros(size(label_test));
    predict_train_LDA_tmp(find(mappedX > 0)) = 2;
    predict_train_LDA_tmp(find(mappedX <= 0)) = 1;
    correct_num_LDA_tmp(i) = sum(predict_train_LDA_tmp == label_train);
    if correct_num_LDA_tmp(i) > correct_num_LDA(i)
        correct_num_LDA(i) = correct_num_LDA_tmp(i);
        predict_train_LDA = predict_train_LDA_tmp;
    end
    pre_train_LDA = [pre_train_LDA;predict_train_LDA];
    test_data = bsxfun(@minus, test_data, mapping.mean);
    mappedtest = test_data * mapping.M;
    predict_test_LDA = zeros(size(label_test));
    predict_test_LDA(find(mappedtest > 0)) = 1;
    predict_test_LDA(find(mappedtest <= 0)) = 2;
    correct_num_LDA(i) = sum(predict_test_LDA == label_test);
    predict_test_LDA_tmp = zeros(size(label_test));
    predict_test_LDA_tmp(find(mappedtest > 0)) = 2;
    predict_test_LDA_tmp(find(mappedtest <= 0)) = 1;
    correct_num_LDA_tmp(i) = sum(predict_test_LDA_tmp == label_test);
    if correct_num_LDA_tmp(i) > correct_num_LDA(i)
        correct_num_LDA(i) = correct_num_LDA_tmp(i);
        predict_test_LDA = predict_test_LDA_tmp;
    end
    pre_LDA = [pre_LDA;predict_test_LDA];
    accuracy_LDA(i) = correct_num_LDA(i)/length(label_test);
    fprintf("LDA Fold No.%d, Correct number: %d, total: %d.\n", i, correct_num_LDA(i), tmp(i));
    train_num = train_num + tmp(i);
end
fprintf("ACC mean: %.4f, std: %.4f\n",mean(accuracy_LDA),std(accuracy_LDA));
[CM_LDA_train,~] = confusionmat(y_train, pre_train_LDA);
fprintf("LDA train:\n");
[acc_train_LDA, prec_train_LDA,sen_train_LDA,spe_train_LDA,f1_train_LDA]...
    = print_performance(CM_LDA_train);
[CM_LDA_test,~] = confusionmat(y_test, pre_LDA);
fprintf("LDA validate:\n");
[acc_test_LDA, prec_test_LDA,sen_test_LDA,spe_test_LDA,f1_test_LDA]...
    = print_performance(CM_LDA_test);
avr_acc_LDA = sum(correct_num_LDA)/length(label);
fprintf("LDA Average Accuracy:%.4f\n",avr_acc_LDA);

function [acc,pre,sen,spe,f1] = print_performance(CM)
    TP = CM(1,1);
    FP = CM(1,2);
    FN = CM(2,1);
    TN = CM(2,2);
    acc = (TP+TN)/sum(sum(CM));
    pre = TP/(TP + FP); % ppv
    sen = TP/(TP + FN); % sensitivity
    spe = TN/(FP + TN); % specificity
    npv = TN/(FN + TN); % npv
    f1 = 2*(pre*sen)/(pre+sen);
    fprintf("Acc: %f , precision(ppv): %f , sensitivity: %f, specificity: %f,NPV：%f，F1-score: %f\n",...
    acc,pre,sen,spe,npv,f1);
end
