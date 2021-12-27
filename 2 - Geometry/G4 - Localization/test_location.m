%% ######## G3 Localization ########
% computes the inverse of the metric homography

H = [-0.9986,    0.1357,   -0.0001;
     -0.0521,   -2.6004,    0.0005;
     -0.0001,    0.0013,    1.0000;];

K = 1.0e+03 * [1.3158, 0,      0.5481;
               0,      0.8163, 0.9504;
               0,      0,      0.0010;]; 
H = inv(H);
           
% extract columns
h1 = H(:,1);
h2 = H(:,2);
h3 = H(:,3);

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

%T = T * 1.5 / T(3);

cameraRotation = R.';
% since T is expressed in the camera ref frame we want it in the plane
% reference frame, R.' is the rotation of the camera wrt the plane
cameraPosition = -R.'*T;


% Display orientation and position from left horizontal face

AA = [0 0];
BB = [0 1000];
CC = [500 0];
DD = [500 1000];

A = 1.0e+03 * [-0.2420   -1.0925];
B = 1.0e+03 * [-0.6367   -1.1191];
C = 1.0e+03 * [-0.1664   -2.0237];
D = 1.0e+03 * [-0.5610   -2.0503];

%AA = [dist([A,C],'euclidean') 0];
%BB = [dist([A,B]) dist([B,D])];
%CC = [0 0];
%DD = [0 5];

face = [[AA(1); BB(1); CC(1); DD(1)],[300; 300; 300; 300], [AA(2); BB(2); CC(2); DD(2)]];

figure
plotCamera('Location', cameraPosition, 'Orientation', cameraRotation.', 'Size', 20);
hold on
pcshow(face, 'green','VerticalAxisDir', 'down', 'MarkerSize', 100);
xlabel('X')
ylabel('Y')
zlabel('Z')
hold off