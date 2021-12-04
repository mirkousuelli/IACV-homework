% reading the image file
I = imread('../../img/affined_img.png');
I = rgb2gray(I);

% find lines 
lines = findLines(I)

% plot lines
fig = plotLines(I, lines, 'red', 'lines')

% show the edge image
figure(1), imshow(fig)