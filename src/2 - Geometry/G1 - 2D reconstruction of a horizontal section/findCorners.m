function [loc_x,loc_y] = findCorners(img, sigma, border_margin)
% HARRIS ALGORITHM: find corner features in an image

    % Derivative masks
    dx = [-1 0 1; 
          -1 0 1; 
          -1 0 1];
    dy = dx';
      
    % Image derivatives : applying the filter through convolution
    Ix = conv2(img, dx, 'same');
    Iy = conv2(img, dy, 'same');

    % Gaussian filter
    g = fspecial('gaussian',max(1,fix(3*sigma)+1), sigma);

    % applying the gaussian filter to images
    Ix2 = conv2(Ix.^2, g, 'same');
    Iy2 = conv2(Iy.^2, g, 'same');
    Ixy = conv2(Ix.*Iy, g, 'same');

    % combining filtered images
    cm = (Ix2.*Iy2 - Ixy.^2)./(Ix2 + Iy2 + eps);

    % Set to 0 near the boundaries
    cm(1:border_margin,:)=0;
    cm(end-border_margin:end,:)=0;
    cm(:,end-border_margin:end)=0;
    cm(:,1:border_margin)=0;

    % Threshold the cim
    T=mean(cm(:));
    CIM=cm;
    CIM(cm<T)=0;

    % perform nonlocal maximum suppression on the thresholded measure
    support=true(11);
    
    % compute maximum over a square neighbor of size 11 x 11
    maxima=ordfilt2(CIM,sum(support(:)),support);
    
    % determine the locations where the max over the neigh or 11x11
    % corresponds to the cim values
    [loc_x,loc_y]=find((cm==maxima).*(CIM>0));
end
