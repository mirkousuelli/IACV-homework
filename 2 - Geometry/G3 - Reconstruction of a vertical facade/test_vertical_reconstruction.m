% camera calibration
% generate the list of perpendicular pairs of vanishing points to pass to 
% the calibration algorithm

%load the image
clc;
clear;
close all;
img = imread('../../img/villa_image.png');
img_gray = rgb2gray(img);

IMG_MAX_SIZE = max((size(img)));
H_scaling = diag([1/IMG_MAX_SIZE, 1/IMG_MAX_SIZE, 1]);

% interactively select f families of segments that are images of 3D parallel lines
lines = findLines(img_gray);

% CONSTRUCTION LINE
idx = [3,  19]; % vertical
label = ['m', 'n'];
fig = figure();
imshow(img), hold on;
for k = 1:length(idx)
    xy = [lines(idx(k)).point1; lines(idx(k)).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color', 'red');
    text(xy(:,1),xy(:,2), label(k), 'FontSize', 20, 'Color', 'red'); 
end

line_m = cross([lines(idx(1)).point1 1]', [lines(idx(1)).point2 1]');
line_m = line_m / line_m(3);
line_n = cross([lines(idx(2)).point1 1]', [lines(idx(2)).point2 1]');
line_n = line_n / line_n(3);
vvz1 = cross(line_m, line_n);
vvz1 = vvz1 / vvz1(3);

% POINTS (CORNER DETECTION)
A = [226 334 1]';
B = [272 534 1]';
E = [731 551 1]';
G = [821 364 1]';

J = [203 1308 1]';
K = [804 1292 1]';

plot(A(1), A(2),'.b','MarkerSize',12);
text(A(1), A(2), 'A', 'FontSize', 24, 'Color', 'b');
plot(B(1), B(2),'.b','MarkerSize',12);
text(B(1), B(2), 'B', 'FontSize', 24, 'Color', 'b');
plot(E(1), E(2),'.b','MarkerSize',12);
text(E(1), E(2), 'E', 'FontSize', 24, 'Color', 'b');
plot(G(1), G(2),'.b','MarkerSize',12);
text(G(1), G(2), 'G', 'FontSize', 24, 'Color', 'b');
plot(J(1), J(2),'.b','MarkerSize',12);
text(J(1), J(2), 'J', 'FontSize', 24, 'Color', 'b');
plot(K(1), K(2),'.b','MarkerSize',12);
text(K(1), K(2), 'K', 'FontSize', 24, 'Color', 'b');

line_ab = cross(A,B);
line_ab = line_ab / line_ab(3);
ab = [[A(1) A(2)]; [B(1) B(2)]];
plot(ab(:,1), ab(:,2), 'LineWidth', 2, 'Color', 'blue');

line_be = cross(B,E);
line_be = line_be / line_be(3);
be = [[B(1) B(2)]; [E(1) E(2)]];
plot(be(:,1), be(:,2), 'LineWidth', 2, 'Color', 'blue');

line_eg = cross(E,G);
line_eg = line_eg / line_eg(3);
eg = [[E(1) E(2)]; [G(1) G(2)]];
plot(eg(:,1), eg(:,2), 'LineWidth', 2, 'Color', 'blue');

line_ga = cross(G,A);
line_ga = line_ga / line_ga(3);
ga = [[G(1) G(2)]; [A(1) A(2)]];
plot(ga(:,1), ga(:,2), 'LineWidth', 2, 'Color', 'blue');

line_bj = cross(B,J);
line_bj = line_bj / line_bj(3);
bj = [[B(1) B(2)]; [J(1) J(2)]];
plot(bj(:,1), bj(:,2), 'LineWidth', 2, 'Color', 'red');

line_ek = cross(E,K);
line_ek = line_ek / line_ek(3);
ek = [[E(1) E(2)]; [K(1) K(2)]];
plot(ek(:,1), ek(:,2), 'LineWidth', 2, 'Color', 'red');

hold off;

vvx = cross(line_ga, line_be);
vvx = vvx / vvx(3);

vvy = cross(line_ab, line_eg);
vvy = vvy / vvy(3);

vvz2 = cross(line_bj, line_ek);
vvz2 = vvz2 / vvz2(3);

inf_line = cross(vvx, vvy);
inf_line = inf_line / inf_line(3);

H_aff = [1.0000, 0,      -0.0001;
         0,      1.0000, -0.0008;
         0,      0,       1.0000;];
     
%H_met = [1.5873, -0.4036, 0;
%        -0.4036,  1.2664, 0;
%         0,       0,      1.0000;];

H_met = [1.1376,   -0.3067,         0;
        -0.3067,    1.7169,         0;
         0,         0,              1.0000;];
     
H = H_met * H_aff;
H = inv(H);

% Using L_inf, vertical vp and homography
IAC = get_IAC(inf_line, vvz1, vvx, vvy, H);

% get the intrinsic parameter before the denormalization
alfa = sqrt(IAC(1,1));
u0 = -IAC(1,3)/(alfa^2);
v0 = -IAC(2,3);
fy = sqrt(IAC(3,3) - (alfa^2)*(u0^2) - (v0^2));
fx = fy / alfa;

% build K using the parametrization
K = [fx 0 u0; 0 fy v0; 0 0 1];

disp(K);

%% ---------------------------------------------------------------------------------------------------------------
AA = [203 1268 1]; % A
BB = [272 534 1]; % B
CC = [731 551 1]; % C
DD = [799 1255 1]; % D

if 2 == 2
    imshow(img), hold on;
    plot(AA(1), AA(2),'.b','MarkerSize',12);
    text(AA(1), AA(2), 'A', 'FontSize', 24, 'Color', 'm');
    plot(BB(1), BB(2),'.b','MarkerSize',12);
    text(BB(1), BB(2), 'B', 'FontSize', 24, 'Color', 'm');
    plot(CC(1), CC(2),'.b','MarkerSize',12);
    text(CC(1), CC(2), 'C', 'FontSize', 24, 'Color', 'm');
    plot(DD(1), DD(2),'.b','MarkerSize',12);
    text(DD(1), DD(2), 'D', 'FontSize', 24, 'Color', 'm');
    hold off;
    pause;
end

line_ab = cross(AA, BB);
line_ab = line_ab / line_ab(3);
line_cd = cross(CC, DD);
line_cd = line_cd / line_cd(3);
VV0 = cross(line_ab, line_cd);
VV0 = VV0 / VV0(3);

line_bc = cross(CC, BB);
line_bc = line_bc / line_bc(3);
line_ad = cross(AA, DD);
line_ad = line_ad / line_ad(3);
VV2 = cross(line_bc, line_ad);
VV2 = VV2 / VV2(3);

inf_line = cross(VV0, VV2);
inf_line = inf_line / inf_line(3);


w = inv(K * K');

%w = IAC;


syms 'x';
syms 'y';

% A  B  D
% B  C  E
% D  E  F
% Ax^2 + 2Bxy + Cy^2 + 2Dx + 2Ey + F = 0

eq1 = w(1,1)*x^2 + 2*w(1,2)*x*y + w(2,2)*y^2 + 2*w(1,3)*x + 2*w(2,3)*y + w(3,3);
eq2 = inf_line(1)*x + inf_line(2) * y + inf_line(3);

eqns = [eq1 == 0, eq2 == 0];
sol = solve(eqns, [x,y]);

%solutions

II = [double(sol.x(1));double(sol.y(1));1];
JJ = [double(sol.x(2));double(sol.y(2));1];

imDCCP = II*JJ.' + JJ*II.';
%imDCCP = H_scaling \ imDCCP;
imDCCP = imDCCP./norm(imDCCP);

%compute the rectifying homography
[U,D,V] = svd(imDCCP);
D(3,3) = 1;
H_vert = inv(U*sqrt(D));

tform = projective2d(H_vert.');
img = imwarp(img,tform);
img = flip(img ,1);   %# vertical flip
img = flip(img,2);    %# horizontal+vertical flip
%img = imrotate(img, 185);

[AA(1),AA(2)] = transformPointsForward(tform,AA(1),AA(2))
[BB(1),BB(2)] = transformPointsForward(tform,BB(1),BB(2))
[CC(1),CC(2)] = transformPointsForward(tform,CC(1),CC(2))
[DD(1),DD(2)] = transformPointsForward(tform,DD(1),DD(2))

AA = - AA(1:2);
BB = - BB(1:2);
CC = - CC(1:2);
DD = - DD(1:2);

AA(2) = AA(2) + 171;
BB(2) = BB(2) + 171;
CC(2) = CC(2) + 171;
DD(2) = DD(2) + 171;

fig = figure();
imshow(img), hold on;
plot(AA(1), AA(2),'.b','MarkerSize',12);
text(AA(1), AA(2), 'A', 'FontSize', 24, 'Color', 'm');
plot(BB(1), BB(2),'.b','MarkerSize',12);
text(BB(1), BB(2), 'B', 'FontSize', 24, 'Color', 'm');
plot(CC(1), CC(2),'.b','MarkerSize',12);
text(CC(1), CC(2), 'C', 'FontSize', 24, 'Color', 'm');
plot(DD(1), DD(2),'.b','MarkerSize',12);
text(DD(1), DD(2), 'D', 'FontSize', 24, 'Color', 'm');
hold off;

short = sqrt((AA(1) - BB(1))^2 + (AA(2) - BB(2))^2)
long = sqrt((AA(1) - CC(1))^2 + (AA(2) - CC(2))^2)
ratio = short / long
