function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

bst_pdt = Inf;%best prediction

for C_tst = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for sigma_tst = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        model = svmTrain(X ,y ,C_tst ,...
            @(x1 ,x2) gaussianKernel(x1 ,x2 ,sigma_tst));
        
        predictions = svmPredict(model ,Xval);
        pdt = mean(double(predictions ~= yval));
        
        if pdt < bst_pdt
            bst_pdt  = pdt;
            C = C_tst;
            sigma = sigma_tst;
        end
    end
end

% =========================================================================

end
