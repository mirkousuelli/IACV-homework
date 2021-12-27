function lines = findLines(img)
% HOUGH TRANSFORMATION: find line features in an image

    % to gray scale
    %gray_image = rgb2gray(image);
    
    % Canny Algorithm for edge detection
    edges = edge(img,'canny', [0.2 0.25]);
    
    % Hough Transformation space
    [H,T,R] = hough(edges, 'RhoResolution', 0.5, 'Theta', -90:0.5:89.5);
    
    % Hough peaks for line detection
    P = houghpeaks(H,50,'threshold',ceil(0.3*max(H(:))), 'NHoodSize', [97 37]);
    
    % line formatting through Hough Transformation
    lines = houghlines(edges,T,R,P,'FillGap',200,'MinLength',500);
end