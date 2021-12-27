R = [1 0 0;
     0 0 -1;
     0 1 0];
t = [10 0 20];
pose = rigid3d(R,t)

cam = plotCamera('AbsolutePose',pose,'Opacity',0)

grid on
axis equal
axis manual

xlim([-15 20]);
ylim([-15 20]);
zlim([15 25])
