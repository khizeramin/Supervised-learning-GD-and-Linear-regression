%D. A second model - polynomial model.
%y1 = [1 x1 x1^2 x1^3] * [a0;a1;a2;a3]
clc
clear all
close all

dataMatrix = load('reg_data_set_1.mat');
P = 15;
% P = 3
X = ones(length(dataMatrix.x),1);
for p = 1:P
    X = [X, dataMatrix.x.^p];
end
w = pinv(X)*dataMatrix.y;

x = 0:.001:1;
y = zeros(1,length(x));
for i = 1:length(x)
    polyTerms = 1;
    for p = 1:P
        polyTerms = [polyTerms, x(i)^p];
    end
    y(i) = polyTerms * w;
        
end



%p = 2;
%x = -1 : .001 :1;
%y = zeros(1, length(x));
%for i = 1: length(x)
   % poly = 1;
    %for j = 1:p
   %     poly = [poly, x(i)^j]; % 1 concat x concat x^2
  %  end
 %   y(i) = poly * w;
%end
fprintf('w0 = %f, w1 = %f, w2 = %f w3 = %f\n',w(1),w(2),w(3),w(4));

plot(dataMatrix.x, dataMatrix.y, '.');
hold on;
plot(x,y, 'r-','linewidth', 2);
title('Polynomial Model - Analytical solution');
xlabel('Feature value x');
ylabel('Output y');
legend('training examples', 'estimated prediction hypothesis(polynomial)')
axis([0, 1, -3, 10])
grid on;