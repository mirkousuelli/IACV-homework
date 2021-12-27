function draw_axis(M, k)
%DRAW_AXIS Given a camera matrix draws the reference frame
arguments
	M (3,4)
	k = 2
end

lw = 5;
pw = 15;

x_axis = M * Seg(HX([0 0 0]), HX(k*[1 0 0]));
y_axis = M * Seg(HX([0 0 0]), HX(k*[0 1 0]));
z_axis = M * Seg(HX([0 0 0]), HX(k*[0 0 1]));

x_axis.draw("Color", "red", "LineWidth", lw);
y_axis.draw("Color", "green", "LineWidth", lw);
z_axis.draw("Color", "blue", "LineWidth", lw);

c = x_axis.line * y_axis.line;
c = c.cart;

% draws a circle in the origin
th = 0:pi/50:2.1*pi;
xunit = 1 * cos(th) + c(1);
yunit = 1 * sin(th) + c(2);
plot(xunit, yunit, "Color", "black", "LineWidth", pw);

end