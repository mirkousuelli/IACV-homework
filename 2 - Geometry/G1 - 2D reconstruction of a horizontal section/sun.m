%% 3D SPACE
PI_1 = [0 0 1 0]';

% corner points
A = [226 334 1 1]';
B = [272 534 1 1]';
%P = [367 537 1 1]';
C = [731 551 1 1]';
D = [821 364 1 1]';

% vanishing points
S = [3.9 -1 1 0]'; % sun direction
S_T = [2.9/3.9 3.9 1 0]'; % orthogonal sun direction

% facade 3
facade_3 = null([B';C';PI_1';]);
facade_3 = facade_3 / facade_3(4);

% vanishing point S ----> corner A
plane_SA = null([S';A';PI_1';]);
plane_SA = plane_SA / plane_SA(4);

% point P on facade 3
% vanishing point S ----> corner A
P = null([facade_3';plane_SA';PI_1';]);
P = P / P(4);




%% 2D PLANE