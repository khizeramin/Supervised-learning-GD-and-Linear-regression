%E. Evaluationg the model.
clc
clear all
close all

dataMatrix = load('reg_data_set_2.mat');

%1) Using the first half of the data set for training and the second half for validation.
train_x = dataMatrix.x(1:length(dataMatrix.x)/2);
val_x = dataMatrix.x(1+length(dataMatrix.x)/2:end);
train_y = dataMatrix.y(1:length(dataMatrix.y)/2);
val_y = dataMatrix.y(1+length(dataMatrix.y)/2:end);
size(dataMatrix.x)
size(val_x)
size(train_y)
size(val_y)
%2)
%P = 4;
for poly = 1:6
   
    %Training set
    TX = ones(length(train_x),1);
    for p = 1:poly
        TX = [TX, train_x.^p];
    end
    %estimating the coefficients of the polynomials.
    w = pinv(TX)*train_y;

    % Theroy for ouput prediction 
    % f(x;w) = X * w 

    train_y_predict = TX * w;
    RMS_train_error = sqrt(sum((train_y_predict - train_y).^2))


    %Validation set 
    VX = ones(length(val_x),1);
    for p = 1:poly
        VX = [VX, val_x.^p];
    end
    val_y_predict = VX * w;
    RMS_val_error = sqrt(sum((val_y_predict - val_y).^2))

    % Plotting the polynomials.
    x = 0:.001:1;
    y = zeros(1,length(x));
    for i = 1:length(x)
        polyTerms = 1;
        for p = 1:poly
            polyTerms = [polyTerms, x(i)^p];   % 1 concat x concat x^2
        end
        y(i) = polyTerms * w;

    end

    
    figure(1);

    subplot(3, 2, poly)
    plot(train_x, train_y, 'b.', 'markersize', 20);
    hold on;
    plot(val_x, val_y, 'g.', 'markersize', 20);
    hold on
    plot(x,y, 'r-','linewidth', 2);
    str_p = int2str(poly);
    title(strcat('Polynomial Model - Analytical solution:  p = ', str_p));
    xlabel('Feature value x');
    ylabel('Output y');
    legend('training examples', 'estimated prediction hypothesis(polynomial)')
    axis([0, 1, -5, 10])
    grid on;
end
