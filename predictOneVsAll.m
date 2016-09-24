function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X.

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
     
prob = sigmoid(X * all_theta');
[predict, prediction] = max(prob, [], 2);
disp(prediction);
p = prediction;

end
