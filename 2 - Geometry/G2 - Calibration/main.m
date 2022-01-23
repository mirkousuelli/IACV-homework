%%
% #########################################################################
% ---------------------------- G2 - CALIBRATION ---------------------------
% #########################################################################

%load the image
clc;
clear;
close all;
img = imread('../../img/villa_image.png');
img_gray = rgb2gray(img);
debug = false;

% lines detection
lines = findLines(img_gray);

% getting two lines to be used to get vertical vanishing point
idx = [3,  19]; % vertical
label = ['m', 'n'];
fig = figure();
imshow(img), hold on;
for k = 1:length(idx)
    xy = [lines(idx(k)).point1; lines(idx(k)).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',4,'Color', 'g');
end

% vertical vanishing point
line_m = cross([lines(idx(1)).point1 1]', [lines(idx(1)).point2 1]');
line_m = line_m / line_m(3);
line_n = cross([lines(idx(2)).point1 1]', [lines(idx(2)).point2 1]');
line_n = line_n / line_n(3);
vvz = cross(line_m, line_n);
vvz = vvz / vvz(3);

% points detected
A = [226 334 1]';
B = [272 534 1]';
E = [731 551 1]';
G = [821 364 1]';

% plotting those points selected
if debug
    plot(A(1), A(2),'.b','MarkerSize',12);
    text(A(1), A(2), 'A', 'FontSize', 24, 'Color', 'b');
    plot(B(1), B(2),'.b','MarkerSize',12);
    text(B(1), B(2), 'B', 'FontSize', 24, 'Color', 'b');
    plot(E(1), E(2),'.b','MarkerSize',12);
    text(E(1), E(2), 'E', 'FontSize', 24, 'Color', 'b');
    plot(G(1), G(2),'.b','MarkerSize',12);
    text(G(1), G(2), 'G', 'FontSize', 24, 'Color', 'b');
end

% plotting linking lines with selected points
line_ab = cross(A,B);
line_ab = line_ab / line_ab(3);
ab = [[A(1) A(2)]; [B(1) B(2)]];
plot(ab(:,1), ab(:,2), 'LineWidth', 4, 'Color', 'r');

line_be = cross(B,E);
line_be = line_be / line_be(3);
be = [[B(1) B(2)]; [E(1) E(2)]];
plot(be(:,1), be(:,2), 'LineWidth', 4, 'Color', 'r');

line_eg = cross(E,G);
line_eg = line_eg / line_eg(3);
eg = [[E(1) E(2)]; [G(1) G(2)]];
plot(eg(:,1), eg(:,2), 'LineWidth', 4, 'Color', 'r');

line_ga = cross(G,A);
line_ga = line_ga / line_ga(3);
ga = [[G(1) G(2)]; [A(1) A(2)]];
plot(ga(:,1), ga(:,2), 'LineWidth', 4, 'Color', 'r');
hold off;

% first horizontal vanishing point
vvx = cross(line_ga, line_be);
vvx = vvx / vvx(3);

% second horizontal vanishing point
vvy = cross(line_ab, line_eg);
vvy = vvy / vvy(3);

% image of the line at the infinity orthogonal to the vertical vanishing
% points in order to obtain two constraints for the calibration process
inf_line = cross(vvx, vvy);
inf_line = inf_line / inf_line(3);

% affine transformation matrix
H_aff = [1.0000, 0,      -0.0001;
         0,      1.0000, -0.0008;
         0,      0,       1.0000;];

% euclidean transformation matrix
H_met = [1.2712,   -0.3197,    0;
        -0.3197,    1.4014,    0;
         0,         0,         1.0000;];

% overall homography to be used for circular points contraints
H = H_met * H_aff;
H = inv(H);

% Using inf_line, vertical vanishing points and homography
IAC = get_IAC(inf_line, vvz, vvx, vvy, H);

% intrinsic parameters
alfa = sqrt(IAC(1,1));
u0 = -IAC(1,3)/(alfa^2);
v0 = -IAC(2,3);
fy = sqrt(IAC(3,3) - (alfa^2)*(u0^2) - (v0^2));
fx = fy / alfa;

% build K using the parametrization
K = [fx 0 u0; 0 fy v0; 0 0 1];

% show calibration matrix
disp(K);