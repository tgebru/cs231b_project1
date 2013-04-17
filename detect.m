function [boxes] = detect(input, model, thresh)

% boxes = detect(input, model, thresh)
% Detect objects in input using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% The first 4 columns specify the bounding box for the root filter and
% subsequent columns specify the bounding boxes of each part.
%
% If bbox is not empty, we pick best detection with significant overlap. 
% If label and fid are included, we write feature vectors to a data file.


% NOTE: You'll need to implement the inference for the part filters in this
% file

% we assume color images
input = color(input);

% prepare model for convolutions
rootfilters = [];
for i = 1:length(model.rootfilters)
  rootfilters{i} = model.rootfilters{i}.w;
end

% cache some data
for c = 1:model.numcomponents
  ridx{c} = model.components{c}.rootindex;
  oidx{c} = model.components{c}.offsetindex;
  root{c} = model.rootfilters{ridx{c}}.w;
  rsize{c} = [size(root{c},1) size(root{c},2)];
end

% we pad the feature maps to detect partially visible objects
padx = ceil(model.maxsize(2)/2+1);
pady = ceil(model.maxsize(1)/2+1);

% the feature pyramid
interval = model.interval;
[feat, scales] = featpyramid(input, model.sbin, interval);

% detect at each scale
best = -inf;
ex = [];
boxes = [];
for level = interval+1:length(feat)
  scale = model.sbin/scales(level);    
  if size(feat{level}, 1)+2*pady < model.maxsize(1) || ...
     size(feat{level}, 2)+2*padx < model.maxsize(2)
    continue;
  end
    
  % convolve feature maps with filters 
  featr = padarray(feat{level}, [pady padx 0], 0);
  rootmatch = fconv(featr, rootfilters, 1, length(rootfilters));
  
  for c = 1:model.numcomponents
    % root score + offset
    score = rootmatch{ridx{c}} + model.offsets{oidx{c}}.w;  
    
    % get all good matches
    I = find(score > thresh);
    [Y, X] = ind2sub(size(score), I);        
    tmp = zeros(length(I), 6);
    for i = 1:length(I)
      x = X(i);
      y = Y(i);
      [x1, y1, x2, y2] = rootbox(x, y, scale, padx, pady, rsize{c});
      b = [x1 y1 x2 y2];
      tmp(i,:) = [b c score(I(i))];
    end
    boxes = [boxes; tmp];
  end
end


% The functions below compute a bounding box for a root or part 
% template placed in the feature hierarchy.
%
% coordinates need to be transformed to take into account:
% 1. padding from convolution
% 2. scaling due to sbin & image subsampling
% 3. offset from feature computation    

function [x1, y1, x2, y2] = rootbox(x, y, scale, padx, pady, rsize)
x1 = (x-padx)*scale+1;
y1 = (y-pady)*scale+1;
x2 = x1 + rsize(2)*scale - 1;
y2 = y1 + rsize(1)*scale - 1;

