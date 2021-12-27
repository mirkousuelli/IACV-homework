function H = homographyEstimation(x, xP)
% Giacomo Boracchi
% course Computer Vision and Pattern Recognition, USI Spring 2020
%
% March 2020
% 
% estimates the homography between pairs of homogeneous points x, xP 

    n = size(x, 2); % number of points -> at least half of the equations required

    %% assemble the design matrix A
    A = zeros(2*n, 9);
    cnt = 1;
    for ii = 1 : n
        p = x(:, ii);
        p = p';
        pP = xP(:, ii);

        A(cnt, :) = [0 0 0, pP(3)*p, 	-pP(2)*p];
        A(cnt + 1, :) = [pP(3)*p, 0 0 0, -pP(1)*p];
        cnt = cnt + 2;
    end

    %% DLT 
    [~, ~, V] = svd(A);
    h = V(:, end); % the last column of V
    H = reshape(h, [3, 3])'; % transpose because h was unrolled row-wise in the derivation in the DLT, while Matlab does reshape column-wise
    H = H / H(3,3);
end