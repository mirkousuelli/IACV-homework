% Giacomo Boracchi
% course Computer Vision and Pattern Recognition, USI Spring 2020
%
% February 2020

close all
clear
clc
I = imread('../../img/villa_image.png');
FNT_SZ = 28;
I = rgb2gray(I); % move to RGB since image transformations have been implented over grayscale images

figure(1), imshow(I);
hold on;

%sun
%a = [320 436 1]';
%b = [290 201 1]';
%c = [231 320 1]';
%d = [100 1369 1]';

a = [224; 1062; 1];
b = [203; 1310; 1];
c = [804; 1293; 1];
d = [783; 1055; 1];
X = [a, b, c, d];

text(a(1), a(2), 'a', 'FontSize', FNT_SZ, 'Color', 'g')
text(b(1), b(2), 'b', 'FontSize', FNT_SZ, 'Color', 'g')
text(c(1), c(2), 'c', 'FontSize', FNT_SZ, 'Color', 'g')
text(d(1), d(2), 'd', 'FontSize', FNT_SZ, 'Color', 'g')

%sun
%aP = [334 527 1]';
%bP = [414 1051 1]';
%cP = [343 1052 1]';
%dP = [336 1304 1]';

aP = [226; 334; 1];
bP = [272; 534; 1];
cP = [731; 551; 1];
dP = [821; 364; 1];
XP = [aP, bP, cP, dP];

text(aP(1), aP(2), 'a*', 'FontSize', FNT_SZ, 'Color', 'm')
text(bP(1), bP(2), 'b*', 'FontSize', FNT_SZ, 'Color', 'm')
text(cP(1), cP(2), 'c*', 'FontSize', FNT_SZ, 'Color', 'm')
text(dP(1), dP(2), 'd*', 'FontSize', FNT_SZ, 'Color', 'm')
hold off;

f = 1; % number of families of parallel lines
numSegmentsPerFamily = 2;
parallelLines = cell(f,1); % store parallel lines
fprintf(['Draw ', num2str(f) , ' families of parallel segments\n']);
col = 'rgbm';
for i = 1:f
    count = 1;
    parallelLines{i} = nan(numSegmentsPerFamily,3);
    while(count <=numSegmentsPerFamily)
        figure(gcf);
        title(['Draw ', num2str(numSegmentsPerFamily),' segments: step ',num2str(count) ]);
        
        if count == 1
            segment1 = drawline('Color',col(1));
            parallelLines{i}(count, :) = segToLine(segment1.Position);
        else
            segment2 = drawline('Color',col(3));
            parallelLines{i}(count, :) = segToLine(segment2.Position);
        end
        count = count +1;
    end
    fprintf('Press enter to continue\n');
    pause;
end

line_a_p1 = [segment1.Position(1,:) 1]';
line_a_p2 = [segment1.Position(2,:) 1]';

line_b_p1 = [segment2.Position(1,:) 1]';
line_b_p2 = [segment2.Position(2,:) 1]';

% estimate homography using DTL algorithm
H = homographyEstimation(X, XP);
%H = H / H(3,3);

line_a_p1 = H * line_a_p1;
line_a_p1 = line_a_p1 / line_a_p1(3);
line_a_p2 = H * line_a_p2;
line_a_p2 = line_a_p2 / line_a_p2(3);

line_b_p1 = H * line_b_p1;
line_b_p1 = line_b_p1 / line_b_p1(3);
line_b_p2 = H * line_b_p2;
line_b_p2 = line_b_p2 / line_b_p2(3);

% Apply the trasformation to the image mapping pixel centers using H^-1 and bilinear interpolation
%J = imwarp%line = line / line(3);Linear(I, H, [1, 1, 1500, 1500]);
figure(2); imshow(I); hold on;

xy = [line_a_p1(1:2)'; line_a_p2(1:2)'];
plot(xy(:,1),xy(:,2),'LineWidth',2,'Color', col(1));
xy = [line_b_p1(1:2)'; line_b_p2(1:2)'];
plot(xy(:,1),xy(:,2),'LineWidth',2,'Color', col(3));

hold off;

%%
function [l] = segToLine(pts)
% convert the endpoints of a line segment to a line in homogeneous
% coordinates.
%
% pts are the endpoits of the segment: [x1 y1;
%                                       x2 y2]

% convert endpoints to cartesian coordinates
a = [pts(1,:)';1];
b = [pts(2,:)';1];
l = cross(a,b);
l = l / l(3);
end
