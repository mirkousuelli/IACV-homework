%%
% #########################################################################
% -------------- G3 - RECONSTRUCTION OF A VERTICAL FACADE -----------------
% #########################################################################

%load the image
clc;
clear;
close all;
img = imread('../../img/villa_image.png');
img_gray = rgb2gray(img);

% calibration matrix from point G2
K = 1.0e+03 * [1.3158, 0,      0.5481;
               0,      0.8163, 0.9504;
               0,      0,      0.0010;]; 

% image points from corner detection
AA = [203 1268 1]; % A
BB = [272 534 1];  % B
CC = [731 551 1];  % C
DD = [799 1255 1]; % D

% showing the vertical plane to be rectified in the image scene
imshow(img), hold on;
% lines
xy = [AA(1:2); BB(1:2)];
plot(xy(:,1),xy(:,2),'LineWidth',4,'Color', 'g');
xy = [BB(1:2); CC(1:2)];
plot(xy(:,1),xy(:,2),'LineWidth',4,'Color', 'g');
xy = [CC(1:2); DD(1:2)];
plot(xy(:,1),xy(:,2),'LineWidth',4,'Color', 'g');
xy = [DD(1:2); AA(1:2)];
plot(xy(:,1),xy(:,2),'LineWidth',4,'Color', 'g');

% points
plot(AA(1), AA(2),'.r','MarkerSize',12);
text(AA(1), AA(2), 'A', 'FontSize', 24, 'Color', 'r');
plot(BB(1), BB(2),'.r','MarkerSize',12);
text(BB(1), BB(2), 'B', 'FontSize', 24, 'Color', 'r');
plot(CC(1), CC(2),'.r','MarkerSize',12);
text(CC(1), CC(2), 'C', 'FontSize', 24, 'Color', 'r');
plot(DD(1), DD(2),'.r','MarkerSize',12);
text(DD(1), DD(2), 'D', 'FontSize', 24, 'Color', 'r');
hold off;

% first vanishing point
line_ab = cross(AA, BB);
line_ab = line_ab / line_ab(3);
line_cd = cross(CC, DD);
line_cd = line_cd / line_cd(3);
VV0 = cross(line_ab, line_cd);
VV0 = VV0 / VV0(3);

% second vanishing point
line_bc = cross(CC, BB);
line_bc = line_bc / line_bc(3);
line_ad = cross(AA, DD);
line_ad = line_ad / line_ad(3);
VV2 = cross(line_bc, line_ad);
VV2 = VV2 / VV2(3);

% image of the line at the infinity
inf_line = cross(VV0, VV2);
inf_line = inf_line / inf_line(3);

% image of the absolute conic through the calibration matrix K
w = inv(K * K');

% setting the system variables
syms 'x';
syms 'y';

% A  B  D
% B  C  E
% D  E  F
% Ax^2 + 2Bxy + Cy^2 + 2Dx + 2Ey + F = 0

% equation of the image absolute conic
eq1 = w(1,1)*x^2 + 2*w(1,2)*x*y + w(2,2)*y^2 + 2*w(1,3)*x + 2*w(2,3)*y + w(3,3);

% equation of the image of the line at the infinity
eq2 = inf_line(1)*x + inf_line(2) * y + inf_line(3);

% solving the system
eqns = [eq1 == 0, eq2 == 0];
sol = solve(eqns, [x,y]);

%solutions (image of circular points)
II = [double(sol.x(1));double(sol.y(1));1];
JJ = [double(sol.x(2));double(sol.y(2));1];

% image of dual conic
imDCCP = II*JJ.' + JJ*II.';
imDCCP = imDCCP./norm(imDCCP);

%compute the rectifying homography
[U,D,V] = svd(imDCCP);
D(3,3) = 1;
H_vert = inv(U * sqrt(D));

% applying the homography to the image
tform = projective2d(H_vert.');
img = imwarp(img,tform);

% since the homography transform the image upside down, we flip it
img = flip(img, 1); % vertical flip
img = flip(img, 2); % horizontal + vertical flip

% we transform points
[AA(1),AA(2)] = transformPointsForward(tform, AA(1), AA(2));
[BB(1),BB(2)] = transformPointsForward(tform, BB(1), BB(2));
[CC(1),CC(2)] = transformPointsForward(tform, CC(1), CC(2));
[DD(1),DD(2)] = transformPointsForward(tform, DD(1), DD(2));

% flipping and shift also the reference point to match the transformation
AA = - AA(1:2);
BB = - BB(1:2);
CC = - CC(1:2);
DD = - DD(1:2);
AA(2) = AA(2) + 171;
BB(2) = BB(2) + 171;
CC(2) = CC(2) + 171;
DD(2) = DD(2) + 171;

% plotting the final outcome
fig = figure();
imshow(img), hold on;

% lines
xy = [AA(1:2); BB(1:2)];
plot(xy(:,1),xy(:,2),'LineWidth',4,'Color', 'g');
xy = [BB(1:2); CC(1:2)];
plot(xy(:,1),xy(:,2),'LineWidth',4,'Color', 'g');
xy = [CC(1:2); DD(1:2)];
plot(xy(:,1),xy(:,2),'LineWidth',4,'Color', 'g');
xy = [DD(1:2); AA(1:2)];
plot(xy(:,1),xy(:,2),'LineWidth',4,'Color', 'g');

% points
plot(AA(1), AA(2),'.r','MarkerSize',12);
text(AA(1), AA(2), 'A', 'FontSize', 24, 'Color', 'r');
plot(BB(1), BB(2),'.r','MarkerSize',12);
text(BB(1), BB(2), 'B', 'FontSize', 24, 'Color', 'r');
plot(CC(1), CC(2),'.r','MarkerSize',12);
text(CC(1), CC(2), 'C', 'FontSize', 24, 'Color', 'r');
plot(DD(1), DD(2),'.r','MarkerSize',12);
text(DD(1), DD(2), 'D', 'FontSize', 24, 'Color', 'r');
hold off;

% ratios useful for point G4
short = sqrt((AA(1) - BB(1))^2 + (AA(2) - BB(2))^2)
long = sqrt((AA(1) - CC(1))^2 + (AA(2) - CC(2))^2)
ratio = short / long