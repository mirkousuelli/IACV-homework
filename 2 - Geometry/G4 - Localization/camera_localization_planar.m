function [R, t] = camera_localization_planar (K, H)
    % takes in input the calibration matrix of the camera, and an
    % homography from the real world to the image. Returns the rotation
    % matrix that express the camera frame orientation with respect the
    % object frame and the position of the optical center in the object
    % frame. The object frame position depends on how the real world points 
    % coordinates are assigned in the fitting of H. 

    % compute the matrix that has as first two columns i and j of the
    % object reference frame expressed in the camera reference, and as 
    % third column the position of the object reference in the camera 
    % reference. This is actually the localization of the object frame in 
    % the camera frame
    % By setting word frame = camera frame, and exploiting the fact that
    % the object is on the horizontal plane, from the Pinhole model we
    % obtain the following
    ijo = K \ H;
    
    % normalize the i and j vectors
    i_norm = norm(ijo(:, 1), 2);
    i = ijo(:, 1) / norm(ijo(:, 1), 2);
    j = ijo(:, 2) / norm(ijo(:, 2), 2);
    k = cross(i, j) / norm(cross(i, j), 2);
    o = ijo(:, 3)/i_norm;

    % The rotation matrix of the camera in the object reference frame is 
    % the transpose of the rotation matrix of the object in the camera
    % frame
    R = [i, j, k]';
    % due to numerical error R might not be a rotation matrix, approximate it
    [U, ~, V] = svd(R);
    R = U * V';
    % compute the translation vector in the object frame
    t = - R * o;
end 