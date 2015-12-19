clc 
clear all 
close all

% B. Analytical solution for the Linear regression method.
dataMatrix = load('reg_data_set_1.mat');


% theory
% y1 = [1 x1] * [w0;w1] = w0 * 1 + x1*w1 (we had to introduce 1 to accomodate
% the multplication inorder to get the vector form of the line equation. 
% y2 = [1 x2] * [w0;w1]
% ...
% ...
% y = X * w
% 
% pinv(X) * y = w

X = [ones(length(dataMatrix.x),1), dataMatrix.x];

w = pinv(X)*dataMatrix.y;


line_x1 = 0;
line_y1 = w(1) + line_x1*w(2);

line_x2 = 1;
line_y2 = w(1) + line_x2*w(2);

fprintf('w0 = %f, w1 = %f \n',w(1),w(2));

plot(dataMatrix.x, dataMatrix.y, '.');
hold on;
plot([line_x1,line_x2], [line_y1,line_y2], 'r-', 'linewidth', 2);
title('Linear Regression - Analytical solution');
xlabel('Feature value x');
ylabel('Output y');
legend('training examples', 'estimated prediction hypothesis(line)')
grid on;

%Plot the data set and the line learned by the model. Does it looks like a good
%linear approximation?
%Yes 