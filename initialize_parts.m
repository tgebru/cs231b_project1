function model=initialize_parts(model,numparts)

% model.partfilters{i}
%   .size
%   .w
%   .blocklabel
% model.offsets{i}
%   .w
%   .blocklabel
% model.components{i}
%   .rootindex
%   .parts{j}
%     .partindex
%     .defindex
%   .offsetindex
%   .dim
%   .numblocks


%model.partfilters{i}
%  .size
%  .w
%  .blocklabel=2
%model.parts{i}
%  .dim
%  .numblocks
%  .offsetindex=1
%  .


%Part components
% parts{j}.sbin
% parts{j}.interval
% parts{j}.blocksizes
% parts{j}.regmult
% parts{j}.learnmult
% parts{j}.maxsize
% parts{j}.minsize
% parts{j}.size %number of HOG cells (which is 2x or just size with respect
% to root filter?
% parts{j}.partindex=location(xj,yj,w,h)
% parts{j}.defindex=[aj,bj] deformation costs initialized to aj(0, 0) and bj =  -(1, 1)
% parts{j}.w
% parts{j}.offsetindex?
% parts{j}.blocklabel?

ridx = model.components{1}.rootindex;
model_size = model.rootfilters{ridx}.size;

%Part area (6a=0.8*root filter area)
part_area = 0.8*prod(model_size*model.sbin)/6;
aspect = model_size(1)/model_size(2);
w_area = sqrt(part_area/aspect);
h_area = w_area*aspect;
h=round(h_area/model.sbin);
w=round(w_area/model.sbin);

part_size = [h w];
zeroed_model = model.rootfilters{1}.w;

for i=1:numparts
    i
    [x,y]=findHighestEnergy(zeroed_model,model_size,part_size);
    model.components{1}.parts{i}.partindex = [x,y,part_size]; %x & y located from root
    model.components{1}.parts{i}.defindex.a=[0 0];
    model.components{1}.parts{i}.defindex.b=[-1 -1];
    model.components{1}.parts{i}.sbin = 0.5*model.sbin;
    %Iniialize part weights
    part_weight_dim = [part_size(1) 2*part_size(2) 31];
    model.components{1}.parts{i}.w= zeros(part_weight_dim);
    root_weights = subarray(model.rootfilters{1}.w, y, y+h-1, x, x+w-1, 0);
    w_part= interp(root_weights(:),2);
    w_part=reshape(w_part, part_weight_dim);
    model.components{1}.parts{i}.w=w_part;
    zeroed_model(y:y+h-1, x:x+w-1,:)=0;
end
end

function [xp,yp]=findHighestEnergy(model_w,model_size,part_size)    
    h=part_size(1);
    w=part_size(2);
    x_limit = model_size(2)-w;
    y_limit = model_size(1)-h;
    max_energy = 0;
    for y=1:h:y_limit
        for x=1:x_limit
            window = subarray(model_w, y, y+h-1, x, x+w-1, 0);
            energy = sum(sum(sum(abs(window).^2))).^(0.5); %change to norm but for some reason error with 3d matricies
            if energy>max_energy
                %check to see if transpose of window or window itself is
                %same energy
                %check for symmetry
                %if subarray(model_w, y, y+h-1, x, ceil(0.5*(x+w-1)), 0)==subarray(model_w, y, y+h-1, ceil(0.5*(x+w-1)), x+w-1, 0);
                
                %Check for symmetric box
                %x1_prime = w/2 + w/2-x2;
                %x2_prime = end-x1;
                %if 
                max_energy=energy;
                xp=x;
                yp=y;
            end
        end
    end
end

