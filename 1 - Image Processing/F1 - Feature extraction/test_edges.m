%% F1 - Feature extraction
% Combining the learned techniques, find edges, corner features and
% straight lines in the image. Then manually select those features and 
% those lines, that are useful for the subsequent steps.

% reading the image file
I = imread('../../img/villa_image.png');

% transform from rgb to grayscale
I = rgb2gray(I);

% set the image values to double
I = im2double(I);

%level = graythresh(I)

% apply Canny algorithm for feature extraction
%edgs = edge(I, 'canny', [0.15 0.5]);

im1 = edge(I, 'canny', [0.1 0.2]);
im2 = edge(I, 'canny', [0.2 0.3]);
im3 = edge(I, 'canny', [0.2 0.4]);
im4 = edge(I, 'canny', [0.2 0.5]);

% show the edge image
figure(1), imshow([I, im1, im2, im3, im4]);

% save the image with edges extracted
%imwrite(edgs, '../img/villa_edges.png')