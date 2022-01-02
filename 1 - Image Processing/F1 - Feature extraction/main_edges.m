%%
% #########################################################################
% ----------------------- F1 - EDGES DETECTION --------------------------
% #########################################################################

% reading the image file
I = imread('../../img/villa_image.png');

% transform from rgb to grayscale
I = rgb2gray(I);

% set the image values to double
I = im2double(I);

% Otsu Method for selecting the threshold
%level = graythresh(I)

% apply Canny Algorithm for feature extraction
edgs = edge(I, 'canny', [0.025 0.05]);

% show the edge image
figure(1), imshow(edgs);

% save the image with edges extracted
%imwrite(edgs, '../img/villa_edges.png')