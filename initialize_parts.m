function model=initialize_parts(model,numparts)

% model.partfilters{i}
%   .size
%   .w
%   .blocklabel
%   .partner & the index of the part which is symmetric to this one
% model.defs{i}
%    .blocklabel
%    .w = ai and bi should be 4 numbers
% model.components{1}.parts{i}
%    .partindex = x,y, partsize
%    .blocksizes= [4 4*h*w*31]

ridx = model.components{1}.rootindex;
model_size = model.rootfilters{ridx}.size;

%Part area (6a=0.8*root filter area)
part_area = 0.8*prod(model_size*model.sbin)/6;
aspect = model_size(1)/model_size(2);
w_area = sqrt(part_area/aspect);
h_area = w_area*aspect;
h=round(h_area/model.sbin);
w=round(w_area/model.sbin);

%Change some model parameters
model.numparts = numparts;
model.components{1}.numblocks= 2*(numparts+1);


part_size = [h w];
%model weights to be zeroed at the location of the part filters
zeroed_model = model.rootfilters{1}.w;

curPart=1;
for i=1:numparts
    curPart;
    [x,y,hasPartner]=findHighestEnergy(zeroed_model,model_size,part_size);
    if (hasPartner==0)
        model.partfilters{curPart}.partner=0;
    end
    model.components{1}.parts{curPart}.partindex = [x,y,part_size]; %x & y located from root
    model.partfilters{curPart}.size=part_size;
    model.partfilters{curPart}.blocklabel=2*curPart+2;
    %width = ceil(model.partfilters{curPart}.size(2)/2);    %get half of the weights
    model.components{1}.parts{curPart}.blocksizes(2)=4*w*h*31; %check if this is right?means we only learn half the weights
    model.lowerbounds{2*curPart+2}=-100*ones(model.components{1}.parts{curPart}.blocksizes(1),1);
    
    model.defs{curPart}=[0 0 -1 -1];
    model.components{1}.parts{curPart}.sbin = 0.5*model.sbin;
    model.components{1}.parts{curPart}.regmult(1)=1;
    model.components{1}.parts{curPart}.regmult(2)=1;
    model.components{1}.parts{curPart}.learnmult(1)=1;
    model.components{1}.parts{curPart}.learnmult(2)=1;
    model.components{1}.parts{curPart}.blocksizes(1)=size(model.defs{curPart});
    model.defs{curPart}.blocklabel=2*curPart+1;
    model.lowerbounds{2*curPart+1}=-100*ones(model.components{1}.parts{2*curPart+1}.blocksizes(2),1);

   
    %Iniialize part weights
    part_weight_dim = [2*part_size(1) 2*part_size(2) 31];
    model.partfilters{curPart}.w= zeros(part_weight_dim);
    root_weights = subarray(model.rootfilters{1}.w, y, y+h-1, x, x+w-1, 0);
    w_part= interp(root_weights(:),4);
    w_part=reshape(w_part, part_weight_dim);
    if (hasPartner==0)
        %model.partfilters{curPart}.w(:,1:width1,:) = f;
        %model.partfilters{curPart}.w(:,width1+1:end,:)= flipfeat(f(:,1:width2,:)); %do flipping of features here
        model.partfilters{curPart}.w=w_part;
    end
    
    %Initialize partner
     if (hasPartner==1)
        model.partfilters{curPart}.partner =curPart+1;
        curPart=curPart+1;   
        %Initialize the rest of the symmetric part
        model.components{1}.parts{curPart}.partindex = [model_size(2)-x+w,y,part_size]; %x & y located from root
        model.partfilters{curPart}.size=part_size;
        model.partfilters{curPart}.blocklabel=2*curPart+2;
        model.components{1}.parts{curPart}.blocksizes(2)=4*w*h*31; %check if this is right?means we only learn half the weights
        model.lowerbounds{2*curPart+2}=-100*ones(model.components{1}.parts{curPart}.blocksizes(1),1);

        model.defs{curPart}=[0 0 -1 -1];
        model.defs{curPart}.blocklabel=2*curPart+1;
        model.components{1}.parts{curPart}.sbin = 0.5*model.sbin;
        model.components{1}.parts{curPart}.regmult(1)=1;
        model.components{1}.parts{curPart}.regmult(2)=1;
        model.components{1}.parts{curPart}.learnmult(1)=1;
        model.components{1}.parts{curPart}.learnmult(2)=1;
        model.components{1}.parts{curPart}.blocksizes(1)=size(model.defs{curPart});
        model.lowerbounds{2*curPart+1}=-100*ones(model.components{1}.parts{2*curPart+1}.blocksizes(2),1);

        %Iniialize part weights
        model.partfilters{curPart}.w=flipfeat(w_part);
        
     end
    if (curPart==numParts)
        break;
    end
    curPart=curPart+1;
end
%Update model dimensions(number of parameters we have to learn (I don't
%know where 2 came from. It was in the modelinit file. 
model.components{1}.dim = 2 + model.blocksizes(1) + model.blocksizes(2)+model.numparts*...
(model.numparts{curPart}.blocksizes(1)+model.numparts{curPart}.blocksizes(2));

function [xp,yp,partner]=findHighestEnergy(model_w,model_size,part_size)    
    h=part_size(1);
    w=part_size(2);
    x_limit = model_size(2)-w;
    y_limit = model_size(1)-h;
    partner=0;
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
                %If self symmetric partner is 0 else partner is 1;
                
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


