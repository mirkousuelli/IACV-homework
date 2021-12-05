% reading the image file
I = imread('../../img/villa_image.png');
I = rgb2gray(I);

% apply Canny algorithm for feature extraction
%fig_edgs = edge(I, 'canny');

% find lines 
lines = findLines(I);
% plot lines
fig_lines = plotLines(I, lines, 'red', 'lines');

% find corners
[loc_x, loc_y] = findCorners(I, 4, 20);
% plot corners
fig_corners = plotCorners(I, loc_x, loc_y, 'green', 'corners');

% show the edge image
figure(1), imshow(fig_corners)