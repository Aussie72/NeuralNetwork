clear all;
close all;
clc;

%% Import data from excel files %%
data = readmatrix("E:\Study\Open Electives\Neural Network and Fuzzy Logic\Assignments\Assignment1\data3.xlsx");
%% normalize data %%
data(:,1:4) = data(:,1:4)./max(data(:,1:4));
%% converting class 1 and 2 to class 0 and 1 respectively for simplification %%
data(:,5) = data(:,5) - 1;
%% randomize Initialization %%
p = randperm(size(data,1));

%% Division of Training data and Test data randomly %%
train_data = [ones(0.6*size(data,1),1) data(p(1:0.6*size(data,1)),:)];
test_data = [ones(0.4*size(data,1),1) data(p(1:0.4*size(data,1)),:)];
clear p data

%% randomly initialize the weight %%
theta = rand(1, size(train_data,2)-1);

%% initialization of learning rate and no. of iterations and later on adjusting it %%
alpha = 0.0001;
iterations = 100;

%% Training output %%
y = train_data(:,6);

%% Logistic Regression Function %%
for i = 1:iterations
    %% sigmoid function %%
    g = sigmoid(theta*train_data(:,1:5)')';
    %% Weight Update Rule %%
    for j = 1:size(train_data,2) - 1
        theta(j) = theta(j) - alpha*sum((y.*(1-g) + (y-1).*g).*train_data(:,j));
    end
end
clear i j g y

%% prediction of class labels of test data %%
y_p = sigmoid(theta*test_data(:,1:5)')';
t = mean(y_p);
y_p = 1*(y_p>t);
y_t = test_data(:,6);
z = size(y_t);

%% Confusion Matrix - TN: True Negative, TP: True Positive, FN: False Negative, FP: False Positive %%
TN = 0;
TP = 0;
FN = 0;
FP = 0;

for i = 1:40
    if y_t(i) == y_p(i) && y_p(i) == 0
        TN = TN +1;
    elseif y_t(i) == y_p(i) && y_p(i) ~= 0
        TP = TP + 1;
    elseif y_t(i) ~= y_p(i) && y_p(i) == 0
        FP = FP + 1;
    else
        FN = FN + 1;
    end
end

%% Calculation of Performance Measuring Terms %%
%% Sensitivity %%
Sensitivity = TP/(TP + FN);
%% Specificity %%
Specificity = TN/(TN + FP);
%% Accuracy %%
Accuracy = (TP + TN)/(TP + TN + FP + FN);


%% sigmoid function %%
function g = sigmoid(z)
    %% compute sigmoid function %%
    g = 1 ./ (1 + exp(-z));
end

%% Cost Function Calculation %%
function J = compute_cost(train_data, y, theta)
    %% Hypothesis %%
    h = logsig(theta*train_data(:,1:5)')';
    %% cost function %%
    J = sum(y'*log(h) + (1-y')*log(1-h));
end