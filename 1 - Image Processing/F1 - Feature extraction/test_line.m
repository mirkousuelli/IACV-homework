% reading the image file
I = imread('../../img/villa_image.png');

lines = findLines(I)

fig = plotLines(I, lines, 'red', 'lines')

% show the edge image
figure(1), imshow(fig)