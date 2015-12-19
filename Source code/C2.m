%C. Linear regression and the descent method.
% In this excersice i am normalizing the gradient.
clc 
clear all
close all

%Loading the dataset.
dataMatrix = load('reg_data_set_1.mat');
%Use as less iteration as possible and get to the correct anlaytical solution. 

%initial point.
N = 500;
w0 = 0;
w1 = 0;
T = 200; %Iterations should be less.
step = 0.1; % stepsize should be tuned to allow less iterations to converge 
% to correct solution.

% Initial cost for all T iterations
J = zeros(1,T); 

for i = 1:T
    fx = w0 + w1 * dataMatrix.x;
    dj = [sum(fx - dataMatrix.y)/N, sum((fx - dataMatrix.y).*dataMatrix.x)/N];
    dj = dj/norm(dj);
    
    w0 = w0 - step * dj(1);
    w1 = w1 - step * dj(2);
    %computing the cost function: J
    J(i) = (1/(2*N)) * sum( (fx - dataMatrix.y).^2 );  
end

fprintf('w0 = %f, w1 = %f \n',w0,w1);

line_x1 = 0;
line_y1 = w0 + line_x1*w1;

line_x2 = 1;
line_y2 = w0 + line_x2*w1;



figure(1);
subplot(1,2,1);
plot(dataMatrix.x, dataMatrix.y, '.');
hold on;
plot([line_x1,line_x2], [line_y1,line_y2], 'r-', 'linewidth', 2);
grid on;
legend('Training Examples', 'Estimated Prediction Hypothesis (Line)', 'location', 'northwest');
title('Linear Regression - Modified Gradient Descent');
xlabel('Feature value x');
ylabel('Output y');

subplot(1,2,2);
plot(1:T, J, 'b-', 'linewidth', 2);
hold on;
plot(1:2:T, J(1:2:end), 'r.', 'markersize', 10);
grid on;
xlabel('Iterations -->');
ylabel('Cost / Mean Squared Error -->');
title('Modified Gradient Descent Convergence Curve');

