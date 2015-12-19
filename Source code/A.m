clc 
clear all 
close all

% A. Data set analysis. 
% Load the dataset and describe the basic properties
% of the data.

% Question block -1 
dataMatrix = load('reg_data_set_1.mat');
% 
plot(dataMatrix.x, dataMatrix.y, '.');
xlabel('Feature value x');
ylabel('True Output y');