%%
% #########################################################################
% ---------- G1 - 2D RECONSTRUCTION OF A HORIZONTAL SECTION ---------------
% #########################################################################

%% AFFINE CONSTRAINTS

%load the image
clc;
clear;
close all;
img = imread('../../img/villa_image.png');
img_gray = rgb2gray(img);
debug = false;

% getting lines from line detector
lines = findLines(img_gray);

% parallel lines to be exploit for affinity constraints
idx = [8, 12, 16, 18];
label = ['a', 'b', 'c', 'd'];
f = 2; % number of families of parallel lines
numSegmentsPerFamily = 2;
parallelLines = cell(f,1); % store parallel lines

% obtaining parallel lines for affinity constraints
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

fig = figure();
imshow(img), hold on;

% plotting lines
for k = 1:length(idx)
    xy = [lines(idx(k)).point1; lines(idx(k)).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',4,'Color', 'red');
    if debug
        text(xy(:,1),xy(:,2), label(k), 'FontSize', 20, 'Color', 'green');
    end
end
hold off;
pause;

%% EUCLIDEAN CONSTRAINTS
fig = figure();
imshow(img), hold on;

% getting points from corner detection
A = [226 334 1]';
B = [272 534 1]';
E = [731 551 1]';
G = [821 364 1]';
P = [507 313 1]';
Q = [507 328 1]';

% top and bottom corners for the vertical shadow
S_TOP = [342 1065 1]';
S_BOT = [343 1300 1]';

% computing the shadow vertical line through corners previously detected
shadow_line = cross(S_TOP, S_BOT);
shadow_line = shadow_line / shadow_line(3);

% computing linking lines bewteen points
line_s = cross(B, E);
line_s = line_s / line_s(3);
line_t = cross(G, E);
line_t = line_t / line_t(3);
line_u = cross(P, Q);
line_u = line_u / line_u(3);

% points from interesected lines
C = cross(shadow_line, line_s);
C = C / C(3);
D = cross(line_u, line_s);
D = D / D(3);

% points plotting
if debug
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
    plot(G(1), G(2),'.b','MarkerSize',12);
    text(G(1), G(2), 'G', 'FontSize', 24, 'Color', 'b');
end


line_ab = [[A(1) A(2)]; [B(1) B(2)]];
plot(line_ab(:,1),line_ab(:,2),'LineWidth',4,'Color', 'r');
line_ab = cross(A, B);
line_ab = line_ab / line_ab(3);

line_bc = [[B(1) B(2)]; [C(1) C(2)]];
plot(line_bc(:,1),line_bc(:,2),'LineWidth',4,'Color', 'r');
line_bc = cross(B, C);
line_bc = line_bc / line_bc(3);

hold off;
pause;

%% AFFINE TRANSFORMATION

% compute the vanishing points
V = nan(2,f);
for i =1:f
    X = parallelLines{i}(:,1:2);
    Y = -parallelLines{i}(:,3);
    V(:,i) = X\Y;
end

% compute the image of the line at infinity
imLinfty = cross([V(1,1) V(2,1) 1], [V(1,2) V(2,2) 1]);%= fitline(V);
imLinfty = imLinfty./(imLinfty(3));

% build the rectification matrix
H = [eye(2),zeros(2,1); imLinfty(:)'];

% rectify the image and show the result
tform = projective2d(H');

% showing the affine transformation matrix
disp('Affine matrix:');
disp(tform.T);

% applying the transformation and crop the image for the sake of simplicity
J = imwarp(img,tform);
J = imcrop(J,[4351 8805 3237 2141]);

% transform point through the affine homography
[A(1),A(2)] = transformPointsForward(tform,A(1),A(2));
[B(1),B(2)] = transformPointsForward(tform,B(1),B(2));
[C(1),C(2)] = transformPointsForward(tform,C(1),C(2));
[D(1),D(2)] = transformPointsForward(tform,D(1),D(2));
[E(1),E(2)] = transformPointsForward(tform,E(1),E(2));
[G(1),G(2)] = transformPointsForward(tform,G(1),G(2));

figure; imshow(J), hold on;

% points
if debug
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
    plot(G(1), G(2),'.b','MarkerSize',12);
    text(G(1), G(2), 'G', 'FontSize', 24, 'Color', 'b');
end

% lines
seg_ab = [[A(1) A(2)]; [B(1) B(2)]];
plot(seg_ab(:,1),seg_ab(:,2),'LineWidth',4,'Color', 'r');
line_ab = cross(A, B);
line_ab = line_ab / line_ab(3);

seg_bc = [[B(1) B(2)]; [C(1) C(2)]];
plot(seg_bc(:,1),seg_bc(:,2),'LineWidth',4,'Color', 'r');
line_bc = cross(B, C);
line_bc = line_bc / line_bc(3);

facade_3 = sqrt((B(1)-E(1))^2 + (B(2)-E(2))^2);
shadow = sqrt((B(1)-C(1))^2 + (B(2)-C(2))^2);

hold off;
pause;

%% STRATIFIED RECTIFICATION
constr = zeros(2,3);

line_bc = cross(B,C); % shadow
line_bc = line_bc / line_bc(3); % shadow
line_ab = cross(A,B); % outgoing wall
line_ab = line_ab / line_ab(3); % outgoing wall
constr(1,:) = [line_ab(1) * line_bc(1), line_ab(1) * line_bc(2) + line_ab(2) * line_bc(1), line_ab(2) * line_bc(2)];

k = (1 / 3.9) ^ 2; % ratio
seg_bc = C - B; % shadow
seg_ab = B - A; % outgoing wall
constr(2,:) = [seg_bc(1) ^ 2 - k * seg_ab(1) ^ 2, 2 * (seg_bc(1) * seg_bc(2) - k * seg_ab(1) * seg_ab(2)), seg_bc(2) ^ 2 - k * seg_ab(2) ^ 2];

% solve the system
[~,~,v] = svd(constr);
s = v(:,end); %[s11,s12,s22];
S = [s(1), s(2); s(2),s(3)];

% compute the rectifying homography
% image of circular points
[U,DD,V] = svd(S);
constr = U*sqrt(DD)*V';

H = eye(3);
H(1,1) = constr(1,1);
H(1,2) = constr(1,2);
H(2,1) = constr(2,1);
H(2,2) = constr(2,2);

Hrect = inv(H);

tform = projective2d(Hrect');
J = imwarp(J,tform);

% showing the last transformation matrix
disp('Euclidean matrix:');
disp(tform.T);

% cropping the image for the sake of simplicity
J = imcrop(J,[0 648 3969 3789]);

% change coordinates to the points
[A(1),A(2)] = transformPointsForward(tform,A(1),A(2));
[B(1),B(2)] = transformPointsForward(tform,B(1),B(2));
[C(1),C(2)] = transformPointsForward(tform,C(1),C(2));
[D(1),D(2)] = transformPointsForward(tform,D(1),D(2));
[E(1),E(2)] = transformPointsForward(tform,E(1),E(2));
[G(1),G(2)] = transformPointsForward(tform,G(1),G(2));

% shifting after cropping to match original position
A(1) = A(1) + 673;
B(1) = B(1) + 673;
C(1) = C(1) + 673;
D(1) = D(1) + 673;
E(1) = E(1) + 673;
G(1) = G(1) + 673;

A(2) = A(2) + 376;
B(2) = B(2) + 376;
C(2) = C(2) + 376;
D(2) = D(2) + 376;
E(2) = E(2) + 376;
G(2) = G(2) + 376;

facade_3 = sqrt((B(1)-E(1))^2 + (B(2)-E(2))^2);
shadow = sqrt((B(1)-C(1))^2 + (B(2)-C(2))^2);

figure; imshow(J), hold on;

% points
if debug
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
    plot(G(1), G(2),'.b','MarkerSize',12);
    text(G(1), G(2), 'G', 'FontSize', 24, 'Color', 'b');
end

% lines
facade_2 = [[A(1) A(2)]; [B(1) B(2)]];
plot(facade_2(:,1),facade_2(:,2),'LineWidth',6,'Color', 'g');

facade_3 = [[B(1) B(2)]; [E(1) E(2)]];
plot(facade_3(:,1),facade_3(:,2),'LineWidth',6,'Color', 'g');

facade_4 = [[E(1) E(2)]; [G(1) G(2)]];
plot(facade_4(:,1),facade_4(:,2),'LineWidth',6,'Color', 'g');
hold off;
 
%normalized coordinates
norm_factor = max((size(img)));
norm_matrix = diag([1/norm_factor, 1/norm_factor, 1]);

% ratio elements
D_n = (D' * norm_matrix)';
E_n = (E' * norm_matrix)';
G_n = (G' * norm_matrix)';

half_f3 = sqrt((D_n(1)-E_n(1))^2 + (D_n(2)-E_n(2))^2);
f4 = sqrt((G_n(1)-E_n(1))^2 + (G_n(2)-E_n(2))^2);

ratio = f4 / (half_f3 * 2);

disp('Ratio facade_2/facade_3:');
disp(ratio);
