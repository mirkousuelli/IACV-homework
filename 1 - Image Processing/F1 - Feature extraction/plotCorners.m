function fig = plotCorners(input_im, loc_x, loc_y, color, figure_title)
    % plot the lines on the input image. lines must be a list of structs 
    % with elements 'point1' and 'point2', as the one returned by the
    % matlab function houghlines
    fig = figure();
    imshow(input_im), hold on
        %xy = [loc_x(k); loc_y(k)];
    plot(loc_y,loc_x,append(color, '+'));
        %text(lines(k).point1(1), lines(k).point1(2), int2str(k), 'FontSize',15, 'Color', 'white'); 
    title(figure_title);
    hold off
end


