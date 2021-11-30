% reading the image file
I = imread('../../img/villa_image.png');
I = rgb2gray(I);

[loc_x, loc_y] = findCorners(I, 4, 20);

fig = plotCorners(I, loc_x, loc_y, 'green', 'corners');

% show the edge image
figure(1), imshow(fig)