AA = [203 1268]; % A
BB = [272 534];  % B
CC = [731 551];  % C
DD = [799 1255]; % D

Hrec = [-0.9986,    0.1357,   -0.0001;
        -0.0521,   -2.6004,    0.0005;
        -0.0001,    0.0013,    1.0000;];

K = 1.0e+03 * [1.3158, 0,      0.5481;
               0,      0.8163, 0.9504;
               0,      0,      0.0010;]; 

tform = projective2d(Hrec.');
[AA(1),AA(2)] = transformPointsForward(tform,AA(1),AA(2));
[BB(1),BB(2)] = transformPointsForward(tform,BB(1),BB(2));
[CC(1),CC(2)] = transformPointsForward(tform,CC(1),CC(2));
[DD(1),DD(2)] = transformPointsForward(tform,DD(1),DD(2));

% ratios
vert_ratio = 0.4234;
horiz_ratio = 0.7976;


LONG_REAL = 934.2631;
SHORT_REAL = LONG_REAL * vert_ratio;
DEPTH_REAL = SHORT_REAL * horiz_ratio;

real_points = [0          0;            % A
               0          LONG_REAL;    % B
               SHORT_REAL LONG_REAL;    % C
               SHORT_REAL 0;];          % D

image_points = [AA; BB; CC; DD];
 
% homography
tform = fitgeotrans(image_points, real_points, 'projective');
H_img_to_world  = (tform.T).';

% localize the camera wrt the left face
H_world_to_img = inv(H_img_to_world * Hrec);
%[R, t] = camera_localization_planar(K, H_world_to_img);

%--------------------------------------------------------------------------
h1 = H_world_to_img(:,1);
h2 = H_world_to_img(:,2);
h3 = H_world_to_img(:,3);

% normalization factor.
lambda = 1 / norm(K \ h1);

% r1 = K^-1 * h1 normalized
r1 = (K \ h1) * lambda;
r2 = (K \ h2) * lambda;
r3 = cross(r1,r2);

% rotation of the world with respect to the camera (R cam -> world)
% where the world in H = inv(H);H = inv(H);this case is the left horizontal face
R = [r1, r2, r3];

% due to noise in the data R may be not a true rotation matrix.
% approximate it through svd, obtaining a orthogonal matrix
[U, ~, V] = svd(R);
R = U * V';

% Compute translation vector. This vector is the position of the plane wrt
% the reference frame of the camera.
T = (K \ (lambda * h3));

cameraRotation = R.';

% since T is expressed in the camera ref frame we want it in the plane
% reference frame, R.' is the rotation of the camera wrt the plane
cameraPosition = -R.' * T;

%--------------------------------------------------------------------------
figure();
facade_3 = [0 0 0; 0 LONG_REAL 0 ; SHORT_REAL 0 0; SHORT_REAL LONG_REAL 0];
facade_2 = [0 0 0; 0 0 DEPTH_REAL ; 0 LONG_REAL DEPTH_REAL; 0 LONG_REAL 0];
facade_4 = [SHORT_REAL 0 0; SHORT_REAL 0 DEPTH_REAL ; SHORT_REAL LONG_REAL DEPTH_REAL; SHORT_REAL LONG_REAL DEPTH_REAL];

facade_3_pos = [SHORT_REAL/2 LONG_REAL/2 0 SHORT_REAL LONG_REAL 10 0 0 0];
facade_2_pos = [0 LONG_REAL/2 DEPTH_REAL/2 10 LONG_REAL DEPTH_REAL 0 0 0];
facade_4_pos = [SHORT_REAL LONG_REAL/2 DEPTH_REAL/2 10 LONG_REAL DEPTH_REAL 0 0 0];
facade_1_pos = [-SHORT_REAL/2 LONG_REAL/2 DEPTH_REAL SHORT_REAL LONG_REAL 10 0 0 0];
facade_5_pos = [3*SHORT_REAL/2 LONG_REAL/2 DEPTH_REAL SHORT_REAL LONG_REAL 10 0 0 0];

hold on;
pcshow(pointCloud(facade_3),'MarkerSize', 800);
pcshowpair(pointCloud(facade_2), pointCloud(facade_4),'MarkerSize', 800);

showShape('cuboid',facade_1_pos, 'Color','green','Opacity',0.5);
showShape('cuboid',facade_3_pos, 'Color','green','Opacity',0.5);
showShape('cuboid',facade_2_pos, 'Color','green','Opacity',0.5);
showShape('cuboid',facade_4_pos, 'Color','green','Opacity',0.5);
showShape('cuboid',facade_5_pos, 'Color','green','Opacity',0.5);

title('camera localized with respect to facade 3');
%axis equal
%axis manual
xlim([-500 900]);
ylim([0 1000]);
zlim([0 1000]);
plotCamera('location', cameraPosition, 'orientation', cameraRotation.', 'size', 20);
hold off;
