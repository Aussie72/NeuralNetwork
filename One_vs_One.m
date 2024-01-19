clear all;
close all;
clc;

%% import data from excel files %%
data = readmatrix("E:\Study\Open Electives\Neural Network and Fuzzy Logic\Assignments\Assignment1\data4.xlsx");
%% normalizing the data %%
%data(:,1:7) = data(:,1:7)./max(data(:,1:7));
%% randomizing the data %%
p = randperm(size(data,1));

%% Dividing and randomizing the dataset into training and test data %%
train_data = data(p(1:0.6*size(data,1)),:);
test_data = data(p(1:0.4*size(data,1)),:);
s = size(train_data);
clear p data

%% Initialization of learning rate and no. of iterations and later on adjusting it to get more accuracy %%
alpha = 0.001;
iterations = 100;

%% test class labels %%
y_test = test_data(:,8);


%% 1 vs 2 %%
%% finding indexes of different class %%
class_1 = find(train_data(:,8) == 1);
class_2 = find(train_data(:,8) == 2);
k = size(class_1);
g = size(class_2);

%% Class Balancing %%
if k(1) > g(1)
    t = g(1);
    class_1 = class_1(1:t,1);
elseif k(1) < g(1)
    t = k(1);
    class_2 = class_2(1:t,1);
else
    t = g(1);
end
d = zeros(t,8);
f = zeros(t,8);
for i = 1:t
    d(i,:) = train_data(class_1(i,1),:);
    f(i,:) = train_data(class_2(i,1),:);
end
train1 = [d; f];
dum = train1;
for i = 1:2*t
     if train1(i,8) == 1
         dum(i,8) = 0;
     else
         dum(i,8) = 1;
     end
end

%% Initializing randomizing weight and assigning training output %%
theta1 = rand(1, s(2)-1);
y1 = dum(:,8);

%% Logistic Regression Function %%
for i = 1:iterations
     %% signmoid function %%
     g = sigmoid(theta1*dum(:,1:7)')';
     %% Weight Update Rule %%
     for j = 1:s(2)-1
         theta1(j) = theta1(j) - alpha*sum((y1.*(1-g) + (y1-1).*g).*dum(:,j));
     end
end
%% 2 vs 3 %%
%% finding indexes of different class %%
class_2 = find(train_data(:,8) == 2);
class_3 = find(train_data(:,8) == 3);
k = size(class_2);
g = size(class_3);

%% Class Balancing %%
if k(1) > g(1)
    t = g(1);
    class_2 = class_2(1:t,1);
elseif k(1) < g(1)
    t = k(1);
    class_3 = class_3(1:t,1);
end
d = zeros(t,8);
f = zeros(t,8);
for i = 1:t
    d(i,:) = train_data(class_2(i,1),:);
    f(i,:) = train_data(class_3(i,1),:);
end
train2 = [d; f];
dum = train2;
for i = 1:2*t
     if train2(i,8) == 2
         dum(i,8) = 0;
     else
         dum(i,8) = 1;
     end
end

%% Initializing randomizing weight and assigning training output %%
theta2 = rand(1, s(2)-1);
y1 = dum(:,8);

%% Logistic Regression Function %%
for i = 1:iterations
     %% sigmoid function %%
     g = logsig(theta2*dum(:,1:7)')';
     %% Weight Update Rule %%
     theta2(j) = theta2(j) - alpha*sum((y1.*(1-g) + (y1-1).*g).*dum(:,j));
end
 

%% 1 vs 3 %%
%% finding indexes of different class %%
class_1 = find(train_data(:,8) == 1);
class_3 = find(train_data(:,8) == 3);
k = size(class_1);
g = size(class_3);

%% Class Balancing %%
if k(1) > g(1)
    t = g(1);
    class_1 = class_1(1:t,1);
elseif k(1) < g(1)
    t = k(1);
    class_3 = class_3(1:t,1);
end
d = zeros(t,8);
f = zeros(t,8);
for i = 1:t
    d(i,:) = train_data(class_1(i,1),:);
    f(i,:) = train_data(class_3(i,1),:);
end
train3 = [d; f];
dum = train3;
for i = 1:2*t
     if train3(i,8) == 1
         dum(i,8) = 0;
     else
         dum(i,8) = 1;
     end
end

%% Initializing randomizing weight and assigning training output %%
theta3 = rand(1, s(2)-1);
y1 = dum(:,8);

for i = 1:iterations
     %% sigmoid function %%
     g = logsig(theta2*dum(:,1:7)')';
     %% Weight Update Rule %%
     for j = 1:s(2)-1
         theta3(j) = theta3(j) - alpha*sum((y1.*(1-g) + (y1-1).*g).*dum(:,j));
     end
end


%% forming and alloting the max y_prediction %%
%% y_prediction of test data %%
y_prediction1 = sigmoid(theta1*test_data(:,1:7)')';
y_prediction2 = sigmoid(theta2*test_data(:,1:7)')';
y_prediction3 = sigmoid(theta3*test_data(:,1:7)')';
a = [y_prediction1 y_prediction2 y_prediction3];

%% cost function history w.r.t. theta values %%
J = zeros(3, 60);
for i = 1:60
    J(1,i) = compute_cost(train_data(i,1:7), theta1);
    J(2,i) = compute_cost(train_data(i,1:7), theta2);
    J(3,i) = compute_cost(train_data(i,1:7), theta3);
end
J = J';
for i = 1:60
    if a(i,1) == a(i,2) || a(i,1) == a(i,3) || a(i,2) == a(i,3) 
        t(i) = max(J(i,:));
    else
        t(i) = max(a(i,:));
    end
end
    
y_prediction = [];
for i = 1:60
    for j = 1:3
        if t(i) == a(i,j) || t(i) == J(i,j)
            y_prediction = [y_prediction; j];
        end
    end
end

%% Performance Measurements %%
%% confusion matrix %%
u = zeros(3,3);
Individual_Accuracy = zeros(1,3);
for i = 1:60
    if y_test(i) == 1 && y_prediction(i) == 1
        u(1,1) = u(1,1) + 1;
    elseif y_test(i) == 1 && y_prediction(i) == 2
        u(1,2) = u(1,2) + 1;
    elseif y_test(i) == 1 && y_prediction(i) == 3
        u(1,3) = u(1,3) + 1;
    elseif y_test(i) == 2 && y_prediction(i) == 1
        u(2,1) = u(2,1) + 1;
    elseif y_test(i) == 2 && y_prediction(i) == 2
        u(2,2) = u(2,2) + 1;
    elseif y_test(i) == 2 && y_prediction(i) == 3
        u(2,3) = u(2,3) + 1;
    elseif y_test(i) == 3 && y_prediction(i) == 1
        u(3,1) = u(3,1) + 1;
    elseif y_test(i) == 3 && y_prediction(i) == 2
        u(3,2) = u(3,2) + 1;
    else
        u(3,3) = u(3,3) + 1;
    end
end

Individual_Accuary(1,1) = u(1,1)/(u(1,1) + u(1,2) + u(1,3));
Individual_Accuary(1,2) = u(2,2)/(u(2,1) + u(2,2) + u(2,3));
Individual_Accuary(1,3) = u(3,3)/(u(3,1) + u(3,2) + u(3,3));

Overall_Accuracy = (u(1,1) + u(2,2) + u(3,3)) / (u(1,1) + u(2,2) + u(3,3) + u(1,2) + u(1,3) + u(2,1) + u(2,3) + u(3,1) + u(3,2));

%% Sigmoid Function %%
function g = sigmoid(z)
    %% compute sigmoid function %%
    g = 1 ./ (1 + exp(-z));
end

%% Calculating Cost Function %%
function J = compute_cost(train_data, theta)
    m = size(train_data);
    %% hypothesis %%
    h = sigmoid(theta*train_data(:,1:7)')';
    %% train class label %%
    y = train_data(:,m(2));
    %% cost function %%
    J = sum(y' * log(h) - (y' - 1) * log(1-h));
end