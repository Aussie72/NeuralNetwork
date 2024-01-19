clear all;
close all;
clc;

%% Import data from excel files %%
data_1 = readmatrix("E:\Study\Open Electives\Neural Network and Fuzzy Logic\Assignments\Assignment1\training_feature_matrix.xlsx");
data_2 = readmatrix("E:\Study\Open Electives\Neural Network and Fuzzy Logic\Assignments\Assignment1\training_output.xlsx");
data_3 = readmatrix("E:\Study\Open Electives\Neural Network and Fuzzy Logic\Assignments\Assignment1\test_feature_matrix.xlsx");
data_4 = readmatrix("E:\Study\Open Electives\Neural Network and Fuzzy Logic\Assignments\Assignment1\test_output.xlsx");

%% distribute the training data into input and output variables %%
%% normalizing training data %%
x_1 = (data_1(:,1) - mean(data_1(:,1)))/std(data_1(:,1));
x_2 = (data_1(:,2) - mean(data_1(:,2)))/std(data_1(:,2));
y = (data_2 - mean(data_2))/std(data_2);
m = size(y);

%% adding a column %%
x_0 = ones(245,1);
x = [x_0 x_1 x_2];

%% initializing random weight matrix and identity matrix %%
%theta = rand(3,1);
I = eye(3, 3);
%% initialising lambda and later on adjusting it to get the lowest MSE%%
lambda = 0.01;

%% Vectorized Linear Regression %%
theta = (inv(x'*x))*x'*y;
J_linear = compute_vectorized_linear_cost(x,y,theta);

%% Vectorized Ridge Regression %%
theta_Ridge = (inv(x'*x + lambda*I))*x'*y;
J_ridge = compute_vectorized_ridge_cost(x, y, theta_Ridge, lambda);

%%Vectorized Least Angle Regression %%
theta_Least_Angle = (inv(x'*x))*(x'*y - lambda/2*sign(theta));
J_least_angle = compute_vectorized_least_angle_cost(x, y, theta_Least_Angle, lambda);

%% distribute the test data into input and output variables %%
%% normalizing test data %%
x_t1 = (data_3(:,1) - mean(data_3(:,1)))/std(data_3(:,1));
x_t2 = (data_3(:,2) - mean(data_3(:,2)))/std(data_3(:,2));
%% no need for output test to normalize %%
y_t = data_4;

%% adding a column %%
x_t0 = ones(104,1);
x_test = [x_t0 x_t1 x_t2];
z = size(y_t);

%% calculating predicted output of test data with the optimized weight matrix %%
y_p = theta(1)*x_test(:,1) + theta(2)*x_test(:,2) + theta(3)*x_test(:,3);

%% denormalizing the predicted test output %%
ypredicted = y_p*std(data_4) + mean(data_4);

%% Calculating the Mean Squared Error %%
MSE = 0;
for i = 1:z(1)
    MSE = MSE + ((ypredicted(i,1)-y_t(i,1))^2)/z(1);
end

%% Cost Function Calculations %%
%% Vectorized Linear Cost %%
function J = compute_vectorized_linear_cost(x, y, theta)
    %% cost function %%
    J = 1/2*(x*theta - y)'*(x*theta - y);
end

%% Vectorized Ridge Cost %%
function J = compute_vectorized_ridge_cost(x,y,theta, lambda)
    %% Cost Function %%
    J = 1/2*((y - x*theta)'*(y - x*theta) + (lambda/2)*(theta'*theta));
end

%% Vectorized Least Angle Cost %%
function J = compute_vectorized_least_angle_cost(x, y, theta, lambda)
    %% Cost Function %%
    J = 1/2*((y - x*theta)'*(y-x*theta) + (lambda/2)*(theta));
end
