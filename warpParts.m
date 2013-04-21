function warped = warpParts(name, model, c, pos)

% warped = warppos(name, model, c, pos)
% Warp positive examples to fit model dimensions.
% Used for training root filters from positive bounding boxes.

part_size = model.partfilters{1}.size;
hogbins  = model.sbin;
part_pixels = part_size * hogbins;
heights = [pos(:).y2]' - [pos(:).y1]' + 1;
widths = [pos(:).x2]' - [pos(:).x1]' + 1;
numpos = length(pos);
numparts = model.numparts;
warped = cell(numpos,numparts);
cropsize = (part_size+2) * hogbins;

for i = 1:numpos
    if mod(i,100)==0
       fprintf('%s: warp parts: %d/%d\n', name, i, numpos);
    end
    for j=1:numparts
        im = color(imread(pos(i).im));
        padx = model.sbin * widths(i) / part_pixels(2);
        pady = model.sbin * heights(i) / part_pixels(1);
        %since part x and y are relative to the root bounding box, 
        part_x1=round(model.components{c}.parts{j}.partindex(1)*hogbins+pos(i).x1); %[x,y,part_size, x_yprime,x_yprime.^2]; 
        part_x2=round(part_x1+part_size(2)*hogbins);
        part_y1=round(model.components{c}.parts{j}.partindex(2)*hogbins+pos(i).y1);
        part_y2=round(part_y1+part_size(1)*hogbins);
        x1 = round(part_x1-padx);
        x2 = round(part_x2+padx);
        y1 = round(part_y1-pady);
        y2 = round(part_y2+pady);
        window = subarray(im, y1, y2, x1, x2, 1);
        warped{i}{j} = imresize(window, cropsize, 'bilinear');
    end
end

