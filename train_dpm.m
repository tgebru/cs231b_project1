function model = train_dpm(name, cachedir, model, pos)

% model = train_dpm(name, cachedir, model, pos)
% Train LSVM. 

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

% Find the positive examples and save them in the data file
fid = fopen(datfile, 'w');
num = poswarp(name, model, 1, pos, fid);

%We probably need to add initial hard negative examples here
%{
%Add hard negative examples
num = num + negHard(name, model, 1, neg, maxnum-num, fid); ?
%}

fclose(fid);
        
% learn model
writeheader(hdrfile, num, labelsize, model);
% reset initial model 
fid = fopen(modfile, 'wb');
fwrite(fid, zeros(sum(model.blocksizes), 1), 'double');
fclose(fid);

% Call the SVM learning code
%Need to call this in a loop 
%for i=1:10
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
    %model=updateDpmAfterSvm(model) %here we call detect.m on all the parts
    %try out different x & ys of parts relative to the root
    %the part x & ys are stored in 
    %model.components{1}.parts{curPart}.partindex=
    %                   [x,y,part_size, x_yprime,x_yprime.^2]; %x & y
    %                   located from root in terms of #of root #of hog
    %                   cells so the actual x & y in pixels would be x*8 &
    %                   y*8
    %                   part_size=[h w] in terms of #hogcells of the model,
    %                   i.e. 8
    %                   xyprime = [x' y'] from scoring function. This is
    %                   what is fed into the SVM xyprime is calculated as 
    %                           w2 = ceil(part_size(2)/2);
    %                           h2 = ceil(part_size(1)/2);
    %                           vi = [x+w2 y-h2];%center coordinates of the box to use for xprime and yprime
    %                           x_yprime=([x y]-2.*[x y]+vi)./part_size; %xprime yprime in the scoring function
    %and update the model with those locations
 %end
    
% compute threshold for high recall
P = find((labels == 1) .* unique);
pos_vals = sort(vals(P));
model.thresh = pos_vals(ceil(length(pos_vals)*0.05));

% cache model
save([cachedir name '_model'], 'model');

% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
function num = poswarp(name, model, c, pos, cachedir, fid) %model now has parts and pos is the new pos we learned (latent root filter bounding box)
numpos = length(pos);

%Save warped so that we don't have to warp every time
try
  load([cachedir cls '_warped_dpm']);
catch
    warped = warppos(name, model, c, pos);
    save([cachedir name '_warped_dpm'], 'warped');
end

warped_parts = warpParts(name, model, c, warped);%pos); %Just crop the warped root to size of parts
ridx = model.components{c}.rootindex;
oidx = model.components{c}.offsetindex;
rblocklabel = model.rootfilters{ridx}.blocklabel;
oblocklabel = model.offsets{oidx}.blocklabel;
dim = model.components{c}.dim;
width1 = ceil(model.rootfilters{ridx}.size(2)/2);
width2 = floor(model.rootfilters{ridx}.size(2)/2);
%partWidth1= ceil(model.partfilters{1}.size(2)/2);
%partWidth2 = floor(model.partfilters{1}.size(2)/2);
partwidth = model.partfilters{1}.size(2);
numparts=model.numparts;
numblocks=model.numblocks;
pixels = model.rootfilters{ridx}.size * model.sbin;
minsize = prod(pixels);
num = 0;
for i = 1:numpos
    if mod(i,100)==0
        fprintf('%s: warped positive: %d/%d\n', name, i, numpos);
    end
    bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];  %this should be the bounding box of the latent root filter that we've learned
    % skip small examples
    if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
      continue
    end    
    %get warped root
    im = warped{i};
    feat = features(im, model.sbin);
    feat(:,1:width2,:) = feat(:,1:width2,:) + flipfeat(feat(:,width1+1:end,:));
    feat = feat(:,1:width1,:);
    fwrite(fid, [1 2*i-1 0 0 0 numblocks dim], 'int32');
    fwrite(fid, [oblocklabel 1], 'single');
    fwrite(fid, rblocklabel, 'single');
    fwrite(fid, feat, 'single'); 
    
    %Get warped parts and write those 
    %the bounding box for part j in image i is warped_parts{i}{j}
    
    for j=1:numparts
        if model.partfilters{j}.fake
            continue;
        end
             feat= features(warped_parts{i}{j}, 0.5*model.sbin);
             partner=model.partfilters{j}.partner;
             if  partner==0
                feat(:,1:partwidth,:) = feat(:,1:partwidth,:) + flipfeat(feat(:,partwidth+1:end,:));
                feat = feat(:,1:partwidth,:);
             else 
                symmetric_feature = flipfeat(feat); %get symmetric features    
                blockNumber1_s=model.defs{partner}.blocklabel;
                blockNumber2_s=model.partfilters{partner}.blocklabel;
                coefficients1_s= model.components{1}.parts{partner}.partindex(end-3:end);%[x,y,part_size, x_yprime,x_yprime.^2];
           
                %write data for symmetric part
                fwrite(fid, blockNumber1_s, 'single');
                fwrite(fid, coefficients1_s, 'single');
                fwrite(fid, blockNumber2_s, 'single');
                fwrite(fid, symmetric_feature, 'single');
            end  
            %write data for part
            blockNumber1=model.defs{j}.blocklabel;
            blockNumber2=model.partfilters{j}.blocklabel;
            coefficients1= model.components{1}.parts{j}.partindex(end-3:end);%[x,y,part_size, x_yprime,x_yprime.^2];
            fwrite(fid, blockNumber1, 'single');
            fwrite(fid, coefficients1, 'single');
            fwrite(fid, blockNumber2, 'single');
            fwrite(fid, feat, 'single');       
    end
      
    %get flipped example
    feat = features(im(:,end:-1:1,:), model.sbin);    
    feat(:,1:width2,:) = feat(:,1:width2,:) + flipfeat(feat(:,width1+1:end,:));
    feat = feat(:,1:width1,:);
    fwrite(fid, [1 2*i 0 0 0 numblocks dim], 'int32');    
    fwrite(fid, [oblocklabel 1], 'single');
    fwrite(fid, rblocklabel, 'single');
    fwrite(fid, feat, 'single');
    
    %get flipped parts
    for k=1:numparts
        if model.partfilters{k}.fake
            continue;
        end
            feat= features(warped_parts{i}{k}(:,end:-1:1,:),0.5*model.sbin);
            partner=model.partfilters{k}.partner;
            if  partner==0
                feat(:,1:partwidth,:) = feat(:,1:partwidth,:) + flipfeat(feat(:,partwidth+1:end,:));
                feat = feat(:,1:partwidth,:);
            else 
                symmetric_feature = flipfeat(feat); %get symmetric features    
                blockNumber1_s=model.defs{partner}.blocklabel;
                blockNumber2_s=model.partfilters{partner}.blocklabel;
                coefficients1_s=model.components{1}.parts{partner}.partindex(end-3:end);%[x,y,part_size, x_yprime,x_yprime.^2];
           
                %write data for symmetric part
                fwrite(fid, blockNumber1_s, 'single');
                fwrite(fid, coefficients1_s, 'single');
                fwrite(fid, blockNumber2_s, 'single');
                fwrite(fid, symmetric_feature, 'single');
            end  
            %write data for part
            blockNumber1=model.defs{k}.blocklabel;
            blockNumber2=model.partfilters{k}.blocklabel;
            coefficients1=model.components{1}.parts{k}.partindex(end-3:end);%[x,y,part_size, x_yprime,x_yprime.^2];
            fwrite(fid, blockNumber1, 'single');
            fwrite(fid, coefficients1, 'single');
            fwrite(fid, blockNumber2, 'single');
            fwrite(fid, feat, 'single');  
    end
    num = num+2;    
end
%{
% get hard negative examples
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
%} 
