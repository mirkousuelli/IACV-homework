function [XCond, T] = precond(X)
% Giacomo Boracchi
% course Computer Vision and Pattern Recognition, USI Spring 2020
%
% February 2020

tx = mean(X(1, :));
ty = mean(X(2, :));

s = mean(std(X, [], 2));

T = [1/s, 0, -tx/s; 0, 1/s, -ty/s; 0, 0, 1];

XCond = T*X;


