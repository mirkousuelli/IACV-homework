% reading the image file
I = imread('../../img/villa_image.png');
I = im2double(I);
I = rgb2gray(I);

% find corners
[loc_x, loc_y] = findCorners(I, 3, 20); %4,20

% plot corners
fig = plotCorners(I, loc_x, loc_y, 'green', 'corners');

% show the edge image
figure(1), imshow(fig)