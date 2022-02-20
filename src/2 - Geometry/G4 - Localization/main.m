%%
% #########################################################################
% ---------------------------- G4 - LOCALIZATION --------------------------
% #########################################################################

% image points
AA = [203 1268]; % A
BB = [272 534];  % B
CC = [731 551];  % C
DD = [799 1255]; % D

% vertical rectifying homography from point G3

H_met = [1.2398,   -0.3136,    0;
        -0.3136,    1.4399,    0;
         0,         0,         1.0000;];
     
H_aff = [1.0000,    0,         -0.0001;
         0,         1.0000,    -0.0008;
         0,         0,          1.0000;];

Hrec = H_met * H_aff;

% calibration matrix from point G2
K = 1.0e+03 * [1.1693,  0,          0.5250;
               0,       0.9424,     0.8602;
               0,       0,          0.0010;];

% rectifying image points
tform = projective2d(Hrec.');
[AA(1),AA(2)] = transformPointsForward(tform,AA(1),AA(2));
[BB(1),BB(2)] = transformPointsForward(tform,BB(1),BB(2));
[CC(1),CC(2)] = transformPointsForward(tform,CC(1),CC(2));
[DD(1),DD(2)] = transformPointsForward(tform,DD(1),DD(2));

% facade 3 side ratios
horiz_ratio = 0.6034;
vert_ratio = 0.5596;

% facade 3 temporary dimensions
LONG_REAL = 500;
SHORT_REAL = LONG_REAL * vert_ratio;
DEPTH_REAL = SHORT_REAL * horiz_ratio;

% real points in the world reference frame
real_points = [0          0;            % A
               0          LONG_REAL;    % B
               SHORT_REAL LONG_REAL;    % C
               SHORT_REAL 0;];          % D

% image points in the image
image_points = [AA; BB; CC; DD];
 
% homography from image reference frame to world reference frame
tform = fitgeotrans(image_points, real_points, 'projective');
H_img_to_world  = (tform.T).';

% homograpgy from world reference frame to image reference frame
H_world_to_img = inv(H_img_to_world * Hrec);

% localization procedure splitting homography columns
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
% where the world in H = inv(H)
R = [r1, r2, r3];

% due to noise in the data R may be not a true rotation matrix.
% approximate it through svd, obtaining an orthogonal matrix
[U, ~, V] = svd(R);
R = U * V';

% Compute translation vector. This vector is the position of the plane wrt
% the reference frame of the camera.
T = (K \ (lambda * h3));

cameraRotation = R.';

% since T is expressed in the camera reference frame we want it in the plane
% reference frame where R.' is the rotation of the camera wrt the plane
cameraPosition = -R.' * T;

% scaling all the space wrt to the height of the camera assumed to be 1.5m,
% thus converted with 150 for the sake of simplicity;
% NOTE: we are woroking on the XY plane, hence the height is Y, not Z
scale_factor = 150 / cameraPosition(2);
cameraPosition = cameraPosition * scale_factor;
LONG_REAL = LONG_REAL * scale_factor;
SHORT_REAL = SHORT_REAL * scale_factor;
DEPTH_REAL = DEPTH_REAL * scale_factor;

% plotting
figure();

% facades clound points in 3D coordinates
facade_3 = [0 0 0; 0 LONG_REAL 0 ; SHORT_REAL 0 0; SHORT_REAL LONG_REAL 0];
facade_2 = [0 0 0; 0 0 DEPTH_REAL ; 0 LONG_REAL DEPTH_REAL; 0 LONG_REAL 0];
facade_4 = [SHORT_REAL 0 0; SHORT_REAL 0 DEPTH_REAL ; SHORT_REAL LONG_REAL DEPTH_REAL; SHORT_REAL LONG_REAL DEPTH_REAL];

% facades shapes parameters
facade_3_pos = [SHORT_REAL/2 LONG_REAL/2 0 SHORT_REAL LONG_REAL 10 0 0 0];
facade_2_pos = [0 LONG_REAL/2 DEPTH_REAL/2 10 LONG_REAL DEPTH_REAL 0 0 0];
facade_4_pos = [SHORT_REAL LONG_REAL/2 DEPTH_REAL/2 10 LONG_REAL DEPTH_REAL 0 0 0];
facade_1_pos = [-SHORT_REAL/2 LONG_REAL/2 DEPTH_REAL SHORT_REAL LONG_REAL 10 0 0 0];
facade_5_pos = [3*SHORT_REAL/2 LONG_REAL/2 DEPTH_REAL SHORT_REAL LONG_REAL 10 0 0 0];

hold on;

% cloud points
pcshow(pointCloud(facade_3),'MarkerSize', 800);
pcshowpair(pointCloud(facade_2), pointCloud(facade_4),'MarkerSize', 800);

% 3D shapes
light_blue = [0.22 0.63 0.96];
showShape('cuboid',facade_1_pos, 'Color', light_blue,'Opacity',0.5);
showShape('cuboid',facade_3_pos, 'Color', light_blue,'Opacity',0.5);
showShape('cuboid',facade_2_pos, 'Color', light_blue,'Opacity',0.5);
showShape('cuboid',facade_4_pos, 'Color', light_blue,'Opacity',0.5);
showShape('cuboid',facade_5_pos, 'Color', light_blue,'Opacity',0.5);

title('camera localized with respect to facade 3 on the XY plane');
xlim([-1000 2000]);
ylim([0 2000]);
zlim([0 3000]);

% camera position and rotation in the world reference frame
plotCamera('location', cameraPosition, 'orientation', cameraRotation.', 'size', 50);
hold off;
