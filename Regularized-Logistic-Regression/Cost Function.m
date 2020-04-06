function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

hyp = sigmoid(X*theta);

t1 = (-y.*(log(hyp)));
t2 = ((1-y).*(log(1-hyp)));

J = ((1/m)*sum(t1-t2))+((lambda/(2*m))*sum(theta.^2))-((lambda/(2*m))*theta(1)^2);

 grad(1) = (1/m)*(X(:,1)'*(hyp-y));                                  % 1 x 1
 grad(2:end) = (1/m)*(X(:,2:end)'*(hyp-y))+(lambda/m)*(theta(2:end));



% =============================================================

end
