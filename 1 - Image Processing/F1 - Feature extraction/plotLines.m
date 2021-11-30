function fig = plotLines(input_im, lines, color, figure_title)
    % plot the lines on the input image. lines must be a list of structs 
    % with elements 'point1' and 'point2', as the one returned by the
    % matlab function houghlines
    fig = figure();
    imshow(input_im), hold on
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1),xy(:,2),'LineWidth',1,'Color', color);
        text(lines(k).point1(1), lines(k).point1(2), int2str(k), 'FontSize',15, 'Color', 'white'); 
        title(figure_title);
    end
end
