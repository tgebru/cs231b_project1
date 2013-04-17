function model = train(name, cachedir, model, pos, neg, latentPositives)

% model = train(name, model, pos, neg, modelType)
% latentPositives==>0 train using warped positives
%                ==>1 train using latent positives
% Train LSVM. (For now it's just an SVM)
% 

%Type of model to train

 if nargin < 5
    latentPositives = 0;
 end

% SVM learning parameters
C = 0.002*model.numcomponents;
J = 1;

maxsize = 2^28;

globals;
hdrfile = [tmpdir name '.hdr'];
datfile = [tmpdir name '.dat'];
modfile = [tmpdir name '.mod'];
inffile = [tmpdir name '.inf'];
lobfile = [tmpdir name '.lob'];

labelsize = 5;  % [label id level x y]
negpos = 0;     % last position in data mining

% approximate bound on the number of examples used in each iteration
dim = 0;
for i = 1:model.numcomponents
  dim = max(dim, model.components{i}.dim);
end
maxnum = floor(maxsize / (dim * 4));

% Reset some of the tempoaray files, just in case
% reset data file
fid = fopen(datfile, 'wb');
fclose(fid);
% reset header file
writeheader(hdrfile, 0, labelsize, model);  
% reset info file
fid = fopen(inffile, 'w');
fclose(fid);
% reset initial model 
fid = fopen(modfile, 'wb');
fwrite(fid, zeros(sum(model.blocksizes), 1), 'double');
fclose(fid);
% reset lower bounds
writelob(lobfile, model)


% Find the positive examples and safe them in the data file
fid = fopen(datfile, 'w');
if (latentPositives)
    try
      load([cachedir name '_train_latentPos'], 'latentPosRoot');
      num = poswarp(name, model, 1, latentPos, fid);
    catch
      latentPosRoot = getBestScoring(name, model, 1, pos);
      save([cachedir name '_train_latentPos'], 'latentPosRoot');
      num = poswarp(name, model, 1, latentPosRoot, fid);
    end
else
    num = poswarp(name, model, 1, pos, fid);
end
  
% Add random negatives
%if (latentPositives)
num = num + negrandom(name, model, 1, neg, maxnum-num, fid);
fclose(fid);
        
% learn model
writeheader(hdrfile, num, labelsize, model);
% reset initial model 
fid = fopen(modfile, 'wb');
fwrite(fid, zeros(sum(model.blocksizes), 1), 'double');
fclose(fid);

% Call the SVM learning code
cmd = sprintf('./learn %.4f %.4f %s %s %s %s %s', ...
              C, J, hdrfile, datfile, modfile, inffile, lobfile);
fprintf('executing: %s\n', cmd);
status = unix(cmd);
if status ~= 0
  fprintf('command `%s` failed\n', cmd);
  keyboard;
end
    
fprintf('parsing model\n');
blocks = readmodel(modfile, model);
model = parsemodel(model, blocks);
[labels, vals, unique] = readinfo(inffile);
    
% compute threshold for high recall
P = find((labels == 1) .* unique);
pos_vals = sort(vals(P));
model.thresh = pos_vals(ceil(length(pos_vals)*0.05));

% cache model
save([cachedir name '_model'], 'model');

% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
function num = poswarp(name, model, c, pos, fid)
numpos = length(pos);
warped = warppos(name, model, c, pos);
ridx = model.components{c}.rootindex;
oidx = model.components{c}.offsetindex;
rblocklabel = model.rootfilters{ridx}.blocklabel;
oblocklabel = model.offsets{oidx}.blocklabel;
dim = model.components{c}.dim;
width1 = ceil(model.rootfilters{ridx}.size(2)/2);
width2 = floor(model.rootfilters{ridx}.size(2)/2);
pixels = model.rootfilters{ridx}.size * model.sbin;
minsize = prod(pixels);
num = 0;
for i = 1:numpos
    if mod(i,100)==0
        fprintf('%s: warped positive: %d/%d\n', name, i, numpos);
    end
    bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];
    % skip small examples
    if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
      continue
    end    
    % get example
    im = warped{i};
    feat = features(im, model.sbin);
    feat(:,1:width2,:) = feat(:,1:width2,:) + flipfeat(feat(:,width1+1:end,:));
    feat = feat(:,1:width1,:);
    fwrite(fid, [1 2*i-1 0 0 0 2 dim], 'int32');
    fwrite(fid, [oblocklabel 1], 'single');
    fwrite(fid, rblocklabel, 'single');
    fwrite(fid, feat, 'single');    
    
    %get flipped example
    feat = features(im(:,end:-1:1,:), model.sbin);    
    feat(:,1:width2,:) = feat(:,1:width2,:) + flipfeat(feat(:,width1+1:end,:));
    feat = feat(:,1:width1,:);
    fwrite(fid, [1 2*i 0 0 0 2 dim], 'int32');
    fwrite(fid, [oblocklabel 1], 'single');
    fwrite(fid, rblocklabel, 'single');
    fwrite(fid, feat, 'single');
    num = num+2;    
end

% get random negative examples
function num = negrandom(name, model, c, neg, maxnum, fid)
numneg = length(neg);
rndneg = floor(maxnum/numneg);
ridx = model.components{c}.rootindex;
oidx = model.components{c}.offsetindex;
rblocklabel = model.rootfilters{ridx}.blocklabel;
oblocklabel = model.offsets{oidx}.blocklabel;
rsize = model.rootfilters{ridx}.size;
width1 = ceil(rsize(2)/2);
width2 = floor(rsize(2)/2);
dim = model.components{c}.dim;
num = 0;
for i = 1:numneg
  if mod(i,100)==0
    fprintf('%s: random negatives: %d/%d\n', name, i, numneg);
  end
  im = color(imread(neg(i).im));
  feat = features(double(im), model.sbin);  
  if size(feat,2) > rsize(2) && size(feat,1) > rsize(1)
    for j = 1:rndneg
      x = random('unid', size(feat,2)-rsize(2)+1);
      y = random('unid', size(feat,1)-rsize(1)+1);
      f = feat(y:y+rsize(1)-1, x:x+rsize(2)-1,:);
      f(:,1:width2,:) = f(:,1:width2,:) + flipfeat(f(:,width1+1:end,:));
      f = f(:,1:width1,:);
      fwrite(fid, [-1 (i-1)*rndneg+j 0 0 0 2 dim], 'int32');
      fwrite(fid, [oblocklabel 1], 'single');
      fwrite(fid, rblocklabel, 'single');
      fwrite(fid, f, 'single');
    end
    num = num+rndneg;
  end
end


