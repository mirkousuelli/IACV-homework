function fig = plotCorners(input_im, loc_x, loc_y, color, figure_title)
    % plot the lines on the input image. lines must be a list of structs 
    % with elements 'point1' and 'point2', as the one returned by the
    % matlab function houghlines
    fig = figure();
    imshow(input_im), hold on
    
    % '+' marker for each corner detected at position (loc_y, loc_x)
    plot(loc_y,loc_x,append(color, '+'));
    title(figure_title);
    hold off
end


