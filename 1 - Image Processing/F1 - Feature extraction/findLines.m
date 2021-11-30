function lines = findLines(image)
    gray_image = rgb2gray(image);
    edges = edge(gray_image,'canny', [0.2 0.3]);
    [H,T,R] = hough(edges, 'RhoResolution', 0.5, 'Theta', -90:0.5:89.5);
    P = houghpeaks(H,25,'threshold',ceil(0.3*max(H(:))), 'NHoodSize', [97 37]);
    lines = houghlines(edges,T,R,P,'FillGap',200,'MinLength',500);
end