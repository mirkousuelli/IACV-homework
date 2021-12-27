function I2 = imwarpLinear(I,H,bb)
% Giacomo Boracchi
% course Computer Vision and Pattern Recognition, USI Spring 2020
%
% Applies an homgraphy (point-trasformation) H to an image I and defines the output image over a grid having size bb
% 
% March 2020
% 

[x,y] = meshgrid(bb(1):bb(3),bb(2):bb(4));

x_vect = x(:);
y_vect = y(:);
x_vect = x_vect';
y_vect = y_vect';

X = [x_vect; y_vect; ones(size(x_vect))]; % make homogeneous
Y = inv(H) * X; % apply inverse tranform

Y = Y ./ repmat(Y(end,:),size(Y,1),1); % projective division

Y = Y(1:end-1, :); % remove last row

xi=reshape(Y(1,:),size(x,1),[]);
yi=reshape(Y(2,:),size(y,1),[]);
I2=interp2(1:size(I,2),1:size(I,1),double(I),xi,yi,'linear',0);

cast(I2,class(I)); % cast I2 to whatever was I

