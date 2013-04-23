function warped = warpParts(name, model, c, pos)

% warped = warppos(name, model, c, pos)
% Warp positive examples to fit model dimensions.
% Used for training root filters from positive bounding boxes.

part_size = model.partfilters{1}.size;
hogbins  = model.sbin;
part_pixels = part_size * hogbins;
width=size(pos{1},2);
height=size(pos{1},1);
mult_y=height/model.rootfilters{1}.size(1); 
mult_x=width/model.rootfilters{1}.size(2); 
padx = model.sbin * width / part_pixels(2);
pady = model.sbin * height/ part_pixels(1);
numpos = length(pos);
numparts = model.numparts;
warped = cell(numpos,numparts);
cropsize = (part_size+1) * hogbins;

for i = 1:numpos
    if mod(i,100)==0
       fprintf('%s: warp parts: %d/%d\n', name, i, numpos);
    end
    for j=1:numparts
        im  = pos{i};
        %since part x and y are relative to the root bounding box, 
        part_x1=model.components{c}.parts{j}.partindex(1)*mult_x;
        part_x2=part_x1+part_size(2)*mult_x;
        part_y1=model.components{c}.parts{j}.partindex(1)*mult_y;
        part_y2=part_y1+part_size(1)*mult_y;
        x1 = round(part_x1-padx);
        x2 = round(part_x2+padx);
        y1 = round(part_y1-pady);
        y2 = round(part_y2+pady);
        window = subarray(im, y1, y2, x1, x2, 1);
        warped{i}{j} = imresize(window, cropsize, 'bilinear');
    end
end

