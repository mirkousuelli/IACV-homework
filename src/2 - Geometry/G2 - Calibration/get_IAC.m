function IAC = get_IAC(l_infs, vps, vp1s, vp2s, H)
%GET_IAC Returns the Image of the absolute conitc
%   l_infs set of imaged line at inf
%   vps set of vanishing points to be used with l_infs
%   vp1s set of vanishing points orthogonal to the point in the
%   corresponding pos in vp2
%   H is the homography in order to estimate the position of circular
%   points
%   IAC is the image of the absolute conic
% Assume w to have this form [a 0 b
%                             0 1 c 
%                             b c d]

% matrix parametrization
syms a b c d;
omega = [a 0 b; 0 1 c; b c d];

X = []; % should be nxm matrix (n is ls size 2, m is 4)
Y = []; % should be n matrix of target values

% first add constraints on l_infs and vps
% 2 constraints for each couple
% [l_inf]x W vp = 0
eqn = []
for ii = 1:size(l_infs,2)
    
    % first compute the element of l
    li = l_infs(:,ii);
    l1 = li(1,1);
    l2 = li(2,1);
    l3 = li(3,1);
  
    % vector product matrix
    lx = [0 -l3 l2; l3 0 -l1; -l2 l1 0];
    
    % get vp
    xi = vps(:,ii);
    
    eqn = [lx(1,:)*omega*xi == 0, lx(2,:)*omega*xi == 0];

end

% cast equations into matrix form
[A,y] = equationsToMatrix(eqn,[a,b,c,d]);
% concatenate matrices
X = [X;double(A)];
Y = [Y;double(y)];

% eqn contains all the equations
eqn = [];
% add constraints on vanishing points
for ii = 1:size(vp1s,2)
    % first compute the element of x
    vi = vp1s(:,ii);
    ui = vp2s(:,ii);
    
    % vp1' W vp2 = 0
    eqn = [eqn, vi.' * omega * ui == 0];

end

if size(eqn,2)>0
    % cast equations into matrix form
    [A,y] = equationsToMatrix(eqn,[a,b,c,d]);
    % concatenate matrices
    X = [X;double(A)];
    Y = [Y;double(y)];
end

% add constraints on homography
if size(H)>0
    % scaling factor needed in order to get an homogeneous matrix
    % get columns
    h1 = H(:,1);
    h2 = H(:,2);

    % first constraint: h1' w h2 = 0
    eq1 = h1.' * omega * h2 == 0;
    % second equation h1'wh1 = h2' w h2
    eq2 = h1.' * omega * h1 == h2.' * omega * h2;

    [A,y] = equationsToMatrix([eq1,eq2],[a,b,c,d]);
    A = double(A);
    y = double(y);

    % concatenate matrices
    X = [X;A];
    Y = [Y;y];
end

% fit a linear model without intercept
lm = fitlm(X,Y, 'y ~ x1 + x2 + x3 + x4 - 1');
% get the coefficients
W = lm.Coefficients.Estimate;

%W = X.'*X \ (X.'*Y)
% image of absolute conic
IAC = double([W(1,1) 0 W(2,1); 0 1 W(3,1); W(2,1) W(3,1) W(4,1)]);

end