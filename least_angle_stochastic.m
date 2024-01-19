clear all;
close all;
clc;

%% import data from excel files %%
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

%% initializing random weight matrix %%
theta = rand(3,1);

%% initialising number of iterations, learning_rate and lambda and later on adjusting it to get the lowest MSE%%
iterations = 900;
alpha = 0.00005;
lambda = 0.03;

%% calling the Least angle Stochastic Gradient Descent function to optimize theta %%
[theta_0, theta, J_history] = least_angle_stochastic_gradient_descent(x, y, theta, alpha, iterations, lambda);

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
 
%% plot %%
%% plotting no. of iterations vs cost function
no_iterations = 1:iterations+1;
plot(no_iterations, J_history);
%% plotting 3D plot of weight 1 vs weight 2 vs cost_function %%
plot3(theta_0(3,:), theta_0(2,:), J_history);

%% Cost Function Calculation %%
function J = compute_least_angle_cost(x, y, theta, lambda)
    %% calculating hypothesis %%
    h = theta(1)*x(:,1) + theta(2)*x(:,2)+ theta(3)*x(:,3);
    %% cost function %%
    J = (1/2)*(sum((h-y).^2) + lambda*sum(theta));
end

%% Least Angle Stochastic Gradient Descent Function %%
function [theta_0, theta, J_history] = least_angle_stochastic_gradient_descent(x, y, theta, alpha, iterations, lambda)
    m = size(y);
    %% initializing theta_0 matrix required for the 3D plot %%
    theta_0 = [theta];
    %% initializing J_history matrix required for the 3D plot %%
    J_history = [compute_ridge_cost(x,y, theta, lambda)];
    %% assigning any random number from 1 to length(x) %%
    j = 50;
    
    %% Weight Update Rule %%
    for i = 1:iterations
        t = randperm(size(x,1),m(1));
        h = theta(1)*x(t(j),1) + theta(2)*x(t(j),2) + theta(3)*x(t(j),3);
        t1 = theta(1) - alpha*sum(h-y) - alpha*lambda/2*sign(theta(1));;
        t2 = theta(2) - alpha*sum(h-y).*x(t(j),2) - alpha*lambda/2*sign(theta(2));;
        t3 = theta(3) - alpha*sum(h-y).*x(t(j),3) - alpha*lambda/2*sign(theta(3)); ;
        theta(1) = t1;
        theta(2) = t2;
        theta(3) = t3;
        theta_0 = [theta_0 theta];
        J_history = [J_history, compute_least_angle_cost(x,y,theta, lambda)];
        %% convergence test - if converged stop the weight update rule %%
        if theta_0(:,i) == theta_0(:,i+1)
            break
        end
    end
end