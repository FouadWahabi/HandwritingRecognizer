function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


sigmo = sigmoid(X * theta);
reg = lambda / (2 * m) * (theta' * theta - theta(1)^2);
J = 1 / m * (-y' * log(sigmo) - (1 - y') * log(1 - sigmo)) + reg;
mask = ones(size(theta));
mask(1) = 0;

grad = 1 / m * X' * (sigmo - y) + lambda / m * (theta .* mask);






% =============================================================

grad = grad(:);

end
