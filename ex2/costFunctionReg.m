function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
theta_1=theta;
theta_1(1)=0; %% set theta_0 to be zero, because we don't penalize on theta_0

J=-(log(sigmoid(X*theta))'*y+log(ones(m,1)-sigmoid(X*theta))'*(ones(m,1)-y))/m+theta_1'*theta_1*lambda/(2*m);
grad=X'*(sigmoid(X*theta)-y)/m+theta_1*lambda/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
