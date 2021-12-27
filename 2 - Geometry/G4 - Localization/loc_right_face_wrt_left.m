function [R, t] = loc_right_face_wrt_left(Hrec, left_face_edges, ...
    right_face_edges, left_face_long_lines, right_face_long_lines)
    % compute the orientation and the position of the right face with
    % respect to the left face. The two reference systems are placed in the
    % bottom left point of the respective faces
    
    % obtain the transformation from the image to the reconstruction on the
    % horizontal plane where the reference system is placed on the left
    % face, the shape are reconstructed and the lenght are the same as in
    % the real world
    H = horizontal_plane_homography(left_face_edges, Hrec);
    
    % transform the lines. The obtained lines are reconstructed and
    % expressed in the reference system of the left face
    left_face_long_lines_rec = left_face_long_lines / H;
    right_face_long_lines_rec = right_face_long_lines / H;
    % average all the angles
    angles = [];
    for ii = 1:size(left_face_long_lines_rec, 1)
        for jj = 1:size(right_face_long_lines_rec, 1)
            l = left_face_long_lines_rec(ii, 1:2);
            m = right_face_long_lines_rec(jj, 1:2);
            angles = [angles, acos(l*m'/(norm(l, 2)*norm(m, 2)))];
        end
    end

    theta = -mean(angles);
    %R = rotz(rad2deg(theta));
    R = [cos(theta), -sin(theta), 0;
        sin(theta), cos(theta), 0;
        0, 0, 1];
    
    % POSITION
    % find the bottom left point of the left face
    bottom_left_right_face_im = cross( right_face_edges.bottom, ...
                                    right_face_edges.left);
    bottom_left_right_face = bottom_left_right_face_im * H';
    bottom_left_right_face = bottom_left_right_face / ...
        bottom_left_right_face (3);
    t = [bottom_left_right_face(1:2)'; 0];
end