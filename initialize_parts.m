function initialize_parts(model,numparts)
% model.sbin
% model.interval
% model.numblocks
% model.numcomponents
% model.blocksizes
% model.regmult
% model.learnmult
% model.maxsize
% model.minsize
% model.rootfilters{i}
%   .size
%   .w
%   .blocklabel
% model.partfilters{i}
%   .w
%   .blocklabel
% model.defs{i}
%   .anchor
%   .w
%   .blocklabel
% model.offsets{i}
%   .w
%   .blocklabel
% model.components{i}
%   .rootindex
%   .parts{j}
%     .partindex=location(xj,yj,w,h)
%     .defindex=[aj,bj] deformation costs initialized to aj(0, 0) and bj =  -(1, 1)
%   .offsetindex
%   .dim
%   .numblocks

for i=1:numparts
    model.components{1}.parts{numparts}.defindex.a=(0,0);
    model.components{1}.parts{numparts}.defindex.b=(0,0);
    model.components{1}.partindex=findHighestEnergy(model.components{1}.

end