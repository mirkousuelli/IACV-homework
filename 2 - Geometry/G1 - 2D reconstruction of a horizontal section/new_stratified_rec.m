%% Affine rectification from fitted vanishing points
% perform an affine rectification of an image. Vanishing points are estimated 
% from families of parallel lines.Then the line at infinity is fitted to the 
% vanishing points. The homography that maps the line at infinity to its
% canonical form: (0: 0: 1) rectify the image up to an affinity transformation.

%load the image
clc;
clear;
close all;
img = imread('../../img/villa_image.png');
img_gray = rgb2gray(img);
%% interactively select f families of segments that are images of 3D parallel lines
lines = findLines(img_gray);

% PARALLEL LINES
idx = [8, 12, 16, 18];
label = ['a', 'b', 'c', 'd']
f = 2; % number of families of parallel lines
numSegmentsPerFamily = 2;
parallelLines = cell(f,1); % store parallel lines

line_a = cross([lines(idx(1)).point1 1]', [lines(idx(1)).point2 1]');
line_a = line_a / line_a(3);
line_b = cross([lines(idx(2)).point1 1]', [lines(idx(2)).point2 1]');
line_b = line_b / line_b(3);
line_c = cross([lines(idx(3)).point1 1]', [lines(idx(3)).point2 1]');
line_c = line_c / line_c(3);
line_d = cross([lines(idx(4)).point1 1]', [lines(idx(4)).point2 1]');
line_d = line_d / line_d(3);

parallelLines{1}(1,:) = line_a;
parallelLines{1}(2,:) = line_d;
parallelLines{2}(1,:) = line_b;
parallelLines{2}(2,:) = line_c;

figure; imshow(img); hold on;
for k = 1:length(idx)
    xy = [lines(idx(k)).point1; lines(idx(k)).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color', 'green');
    text(xy(:,1),xy(:,2), label(k), 'FontSize', 20, 'Color', 'green');
end

% POINTS (CORNER DETECTION)
A = [226; 334; 1];
B = [272; 534; 1];
C = [731; 551; 1];
D = [821; 364; 1];
X = [A, B, C, D];

a = [224; 1062; 1];
b = [203; 1310; 1];
c = [804; 1293; 1];
d = [783; 1055; 1];
x = [a, b, c, d];

P = [507 313 1]';
Q = [507 328 1]';

H = homographyEstimation(x, X);

pp1_1 = [471 1119 1]';
pp1_2 = [498 1148 1]';
pp2_1 = [547 1118 1]';
pp2_2 = [518 1148 1]';

PP1_1 = H * pp1_1;
PP1_1 = PP1_1 / PP1_1(3);
PP1_2 = H * pp1_2;
PP1_2 = PP1_2 / PP1_2(3);
PP2_1 = H * pp2_1;
PP2_1 = PP2_1 / PP2_1(3);
PP2_2 = H * pp2_2;
PP2_2 = PP2_2 / PP2_2(3);

% LINES FROM CORNER POINTS
line_pp1 = cross(PP1_1, PP1_2);
line_pp1 = line_pp1 / line_pp1(3);
line_pp2 = cross(PP2_1, PP2_2);
line_pp2 = line_pp2 / line_pp2(3);
line_s = cross(B, C);
line_s = line_s / line_s(3);
line_t = cross(C, D);
line_t = line_t / line_t(3);
line_u = cross(P, Q);
line_u = line_u / line_u(3);

% POINTS FROM INTERSECTED LINES
E = cross(line_u, line_s);
E = E / E(3);

plot(A(1), A(2),'.b','MarkerSize',12);
text(A(1), A(2), 'A', 'FontSize', 24, 'Color', 'b');
plot(B(1), B(2),'.b','MarkerSize',12);
text(B(1), B(2), 'B', 'FontSize', 24, 'Color', 'b');
plot(C(1), C(2),'.b','MarkerSize',12);
text(C(1), C(2), 'C', 'FontSize', 24, 'Color', 'b');
plot(D(1), D(2),'.b','MarkerSize',12);
text(D(1), D(2), 'D', 'FontSize', 24, 'Color', 'b');
plot(E(1), E(2),'.b','MarkerSize',12);
text(E(1), E(2), 'E', 'FontSize', 24, 'Color', 'b');

% PERPENDICULARS
PP1 = [[PP1_1(1) PP1_1(2)]; [PP1_2(1) PP1_2(2)]];
plot(PP1(:,1),PP1(:,2),'LineWidth',2,'Color', 'm');
text(PP1(:,1),PP1(:,2), 'l', 'FontSize', 20, 'Color', 'm');

PP2 = [[PP2_1(1) PP2_1(2)]; [PP2_2(1) PP2_2(2)]];
plot(PP2(:,1),PP2(:,2),'LineWidth',2,'Color', 'm');
text(PP2(:,1),PP2(:,2), 'm', 'FontSize', 20, 'Color', 'm');

line_s = [[B(1) B(2)]; [C(1) C(2)]];
plot(line_s(:,1),line_s(:,2),'LineWidth',2,'Color', 'm');
text(line_s(:,1),line_s(:,2), 's', 'FontSize', 20, 'Color', 'm');
line_s = cross(B, C);
line_s = line_s / line_s(3);

line_t = [[C(1) C(2)]; [D(1) D(2)]];
plot(line_t(:,1),line_t(:,2),'LineWidth',2,'Color', 'm');
text(line_t(:,1),line_t(:,2), 't', 'FontSize', 20, 'Color', 'm');
line_t = cross(C, D);
line_t = line_t / line_t(3);
hold off;
pause;

%% compute the vanishing points
V = nan(2,f);
for i =1:f
    X = parallelLines{i}(:,1:2);
    Y = -parallelLines{i}(:,3);
    V(:,i) = X\Y;
end

%% compute the image of the line at infinity
imLinfty = cross([V(1,1) V(2,1) 1], [V(1,2) V(2,2) 1]);%= fitline(V);
imLinfty = imLinfty./(imLinfty(3));

%figure;
%hold all;
%for i = 1:f
%    plot(V(1,i),V(2,i),'o','Color','green','MarkerSize',20,'MarkerFaceColor','green');
%end
%hold all;
%figure; imshow(img);

%% build the rectification matrix

H = [eye(2),zeros(2,1); imLinfty(:)'];
% we can check that H^-T* imLinfty is the line at infinity in its canonical
% form:
%fprintf('The vanishing line is mapped to:\n');
%disp(inv(H)'*imLinfty'); % moved in its canonical position

%% rectify the image and show the result

tform = projective2d(H');

disp('Affine matrix:');
disp(tform.T);

J = imwarp(img,tform);
J = imcrop(J,[4351 8805 3237 2141]);

%imwrite(J,'../../img/affined_img.png');

%% projectivity on points

[A(1),A(2)] = transformPointsForward(tform,A(1),A(2));
[B(1),B(2)] = transformPointsForward(tform,B(1),B(2));
[C(1),C(2)] = transformPointsForward(tform,C(1),C(2));
[D(1),D(2)] = transformPointsForward(tform,D(1),D(2));
[E(1),E(2)] = transformPointsForward(tform,E(1),E(2));
[PP1_1(1),PP1_1(2)] = transformPointsForward(tform,PP1_1(1),PP1_1(2));
[PP1_2(1),PP1_2(2)] = transformPointsForward(tform,PP1_2(1),PP1_2(2));
[PP2_1(1),PP2_1(2)] = transformPointsForward(tform,PP2_1(1),PP2_1(2));
[PP2_2(1),PP2_2(2)] = transformPointsForward(tform,PP2_2(1),PP2_2(2));

figure; imshow(J), hold on;
% POINTS
plot(A(1), A(2),'.b','MarkerSize',12);
text(A(1), A(2), 'A', 'FontSize', 24, 'Color', 'b');
plot(B(1), B(2),'.b','MarkerSize',12);
text(B(1), B(2), 'B', 'FontSize', 24, 'Color', 'b');
plot(C(1), C(2),'.b','MarkerSize',12);
text(C(1), C(2), 'C', 'FontSize', 24, 'Color', 'b');
plot(D(1), D(2),'.b','MarkerSize',12);
text(D(1), D(2), 'D', 'FontSize', 24, 'Color', 'b');
plot(E(1), E(2),'.b','MarkerSize',12);
text(E(1), E(2), 'E', 'FontSize', 24, 'Color', 'b');

% LINES
PP1 = [[PP1_1(1) PP1_1(2)]; [PP1_2(1) PP1_2(2)]];
plot(PP1(:,1),PP1(:,2),'LineWidth',2,'Color', 'm');
text(PP1(:,1),PP1(:,2), 'l', 'FontSize', 20, 'Color', 'm');
line_l = cross(PP1_1, PP1_2);
line_l = line_l / line_l(3);

PP2 = [[PP2_1(1) PP2_1(2)]; [PP2_2(1) PP2_2(2)]];
plot(PP2(:,1),PP2(:,2),'LineWidth',2,'Color', 'm');
text(PP2(:,1),PP2(:,2), 'm', 'FontSize', 20, 'Color', 'm');
line_m = cross(PP2_1, PP2_2);
line_m = line_m / line_m(3);

line_s = [[B(1) B(2)]; [C(1) C(2)]];
plot(line_s(:,1),line_s(:,2),'LineWidth',2,'Color', 'm');
text(line_s(:,1),line_s(:,2), 's', 'FontSize', 20, 'Color', 'm');
line_s = cross(B, C);
line_s = line_s / line_s(3);

line_t = [[C(1) C(2)]; [D(1) D(2)]];
plot(line_t(:,1),line_t(:,2),'LineWidth',2,'Color', 'm');
text(line_t(:,1),line_t(:,2), 't', 'FontSize', 20, 'Color', 'm');
line_t = cross(C, D);
line_t = line_t / line_t(3);
hold off;
pause;

%% RECTIFICATION
constr = zeros(2,3);
constr(1,:) = [line_l(1)*line_m(1),line_l(1)*line_m(2)+line_l(2)*line_m(1), line_l(2)*line_m(2)];
constr(2,:) = [line_t(1)*line_s(1),line_t(1)*line_s(2)+line_t(2)*line_s(1), line_t(2)*line_s(2)];

% solve the system

%S = [x(1) x(2); x(2) 1];
[~,~,v] = svd(constr);
s = v(:,end); %[s11,s12,s22];
S = [s(1),s(2); s(2),s(3)];

% compute the rectifying homography

% image of circular points
imDCCP = [S,zeros(2,1); zeros(1,3)]; % the image of the circular points
[U,DD,V] = svd(S);
constr = U*sqrt(DD)*V';
H = eye(3);
H(1,1) = constr(1,1);
H(1,2) = constr(1,2);
H(2,1) = constr(2,1);
H(2,2) = constr(2,2);

Hrect = inv(H);
Cinfty = [eye(2),zeros(2,1);zeros(1,3)];

tform = projective2d(Hrect');
J = imwarp(J,tform);

disp('Metric matrix:');
disp(tform.T);

J = imcrop(J,[0 648 3969 3789]);

[A(1),A(2)] = transformPointsForward(tform,A(1),A(2));
[B(1),B(2)] = transformPointsForward(tform,B(1),B(2));
[C(1),C(2)] = transformPointsForward(tform,C(1),C(2));
[D(1),D(2)] = transformPointsForward(tform,D(1),D(2));
[E(1),E(2)] = transformPointsForward(tform,E(1),E(2));

A(1) = A(1) + 873;
B(1) = B(1) + 873;
C(1) = C(1) + 873;
D(1) = D(1) + 873;
E(1) = E(1) + 873;

A(2) = A(2) + 656;
B(2) = B(2) + 656;
C(2) = C(2) + 656;
D(2) = D(2) + 656;
E(2) = E(2) + 656;

figure; imshow(J), hold on;

% POINTS
if 2 == 2
    plot(A(1), A(2),'.b','MarkerSize',12);
    text(A(1), A(2), 'A', 'FontSize', 24, 'Color', 'b');
    plot(B(1), B(2),'.b','MarkerSize',12);
    text(B(1), B(2), 'B', 'FontSize', 24, 'Color', 'b');
    plot(C(1), C(2),'.b','MarkerSize',12);
    text(C(1), C(2), 'C', 'FontSize', 24, 'Color', 'b');
    plot(D(1), D(2),'.b','MarkerSize',12);
    text(D(1), D(2), 'D', 'FontSize', 24, 'Color', 'b');
    plot(E(1), E(2),'.b','MarkerSize',12);
    text(E(1), E(2), 'E', 'FontSize', 24, 'Color', 'b');
end

% LINES
facade_2 = [[A(1) A(2)]; [B(1) B(2)]];
plot(facade_2(:,1),facade_2(:,2),'LineWidth',3,'Color', 'red');
text(facade_2(:,1),facade_2(:,2), '2', 'FontSize', 20, 'Color', 'red');

facade_3 = [[B(1) B(2)]; [C(1) C(2)]];
plot(facade_3(:,1),facade_3(:,2),'LineWidth',3,'Color', 'red');
text(facade_3(:,1),facade_3(:,2), '3', 'FontSize', 20, 'Color', 'red');

facade_4 = [[C(1) C(2)]; [D(1) D(2)]];
plot(facade_4(:,1),facade_4(:,2),'LineWidth',3,'Color', 'red');
text(facade_4(:,1),facade_4(:,2), '4', 'FontSize', 20, 'Color', 'red');
hold off;
 
%normalized coordinates
norm_factor = max((size(img)));
norm_matrix = diag([1/norm_factor, 1/norm_factor, 1]);

C_n = (C' * norm_matrix)';
D_n = (D' * norm_matrix)';
E_n = (E' * norm_matrix)';

half_f3 = sqrt((E_n(1)-C_n(1))^2 + (E_n(2)-C_n(2))^2);
f4 = sqrt((C_n(1)-D_n(1))^2 + (C_n(2)-D_n(2))^2);

ratio = f4 / (half_f3 * 2);

disp('Ratio facade_2/facade_3:');
disp(ratio);
