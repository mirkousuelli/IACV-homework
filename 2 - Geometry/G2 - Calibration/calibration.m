function K = calibration (perp_pairs)
    % The im_size argument is used to work in normalized image coordinates,
    % it could be size(im) or a scalar.
    % perp_pairs is a list of structs with fields 'v1', 'v2' that are
    % vanishing points corresponding to perpendicular directions

    syms a b c d

    W_sym = [a, 0, b;
             0, 1, c;
             b, c, d];

    % generate the list of equations
    eq = [];
    for ii = 1 : length(perp_pairs)
        v1 = perp_pairs(ii).v1;
        v2 = perp_pairs(ii).v2;
        eq = [  eq ...
                v1 * W_sym *v2' == 0];
    end
    
    [X, t] = equationsToMatrix(eq, [a, b, c, d]);

    X = double(X);
    t = double(t);
    vals = (X' * X) \ (X' * t);
    IAC_n = [vals(1),         0,      vals(2);
                   0,         1,      vals(3);
             vals(2),   vals(3),      vals(4)];

    a = sqrt(IAC_n(1, 1));
    u = -IAC_n(1, 3)/IAC_n(1, 1);
    v = -IAC_n(2, 3);
    fy = sqrt(IAC_n(3, 3) - IAC_n(1, 1)*u^2 - v^2);
    fx = fy / a;

    K = [fx, 0, u;
         0, fy, v;
         0,  0, 1];
end