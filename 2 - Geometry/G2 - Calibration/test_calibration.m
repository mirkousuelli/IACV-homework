% camera calibration
% generate the list of perpendicular pairs of vanishing points to pass to 
% the calibration algorithm

%load the image
clc;
clear;
close all;
img = imread('../../img/villa_image.png');
img_gray = rgb2gray(img);

%norm_factor = max((size(img)));
%norm_matrix = diag([1/norm_factor, 1/norm_factor, 1]);

% interactively select f families of segments that are images of 3D parallel lines
lines = findLines(img_gray);

% CONSTRUCTION LINES
idx = [3, 19];
label = ['n', 'r']
%fig = figure();
%imshow(img), hold on;
%for k = 1:length(idx)
%    xy = [lines(idx(k)).point1; lines(idx(k)).point2];
%    plot(xy(:,1),xy(:,2),'LineWidth',1,'Color', 'red');
%    text(xy(:,1),xy(:,2), label(k), 'FontSize', 20, 'Color', 'red'); 
%end
%hold off;
line_n = cross([lines(idx(1)).point1 1]', [lines(idx(1)).point2 1]');
line_n = line_n / line_n(3);
line_r = cross([lines(idx(2)).point1 1]', [lines(idx(2)).point2 1]');
line_r = line_r / line_r(3);
V0 = cross(line_n, line_r);
V0 = V0 / V0(3);
V0 = V0';% * norm_matrix;

A = [226 334 1];%; * norm_matrix;
B = [272 534 1];%; * norm_matrix;
E = [731 551 1];%; * norm_matrix;
G = [821 364 1];%; * norm_matrix;

line_t1 = cross(A, B);
line_t1 = line_t1 / line_t1(3);
line_t2 = cross(G, E);
line_t2 = line_t2 / line_t2(3);
V1 = cross(line_t1, line_t2);
V1 = V1 / V1(3);

line_s1 = cross(A, G);
line_s1 = line_s1 / line_s1(3);
line_s2 = cross(B, E);
line_s2 = line_s2 / line_s2(3);
V2 = cross(line_s1, line_s2);
V2 = V2 / V2(3);

V_line = cross(V1, V2);
V_line = V_line / V_line(3);

H_aff = [1.0000, 0,      -0.0001;
         0,      1.0000, -0.0008;
         0,      0,       1.0000;];
     
H_met = [1.2256,-0.3136, 0;
        -0.3136, 1.4664, 0;
         0,      0,      1.0000;];
     
H = H_met * H_aff;
H_inv = inv(H);

% Using L_inf, vertical vp and homography
IAC = get_IAC(V_line', V0', [], [], H_inv);

% get the intrinsic parameter before the denormalization
alfa = sqrt(IAC(1,1));
u0 = -IAC(1,3)/(alfa^2);
v0 = -IAC(2,3);
fy = sqrt(abs(IAC(3,3) - (alfa^2)*(u0^2) - (v0^2)));
fx = fy /alfa;

% build K using the parametrization
K = [fx 0 u0; 0 fy v0; 0 0 1];

disp(K);