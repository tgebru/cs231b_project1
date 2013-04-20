function boxes= getBestScoringBox(name, model, c,pos,saveDir)
% bestScoringIm = getBestScoring(name, model, c, pos)
% construct positive cropped image from window that scores the best with 
% trained root parameteres (i.e. where F.G is max where Fo are root weights
% and G are hog features.
% Used for training root filters from positive bounding boxes.

ridx = model.components{c}.rootindex;
root_size = model.rootfilters{ridx}.size;

numpos = length(pos);

min_overlap = 0.5;
%numpos=2;
boxes=[];

    for i = 1:numpos
        %if mod(i,100)==0
            fprintf('%s: latent position: %d/%d\n', name, i, numpos);
        %end
        im = color(imread(pos(i).im));
 
        detectedBoxes = detect(im, model, model.thresh);
        
        %draw boxes to see on image
       %{ 
        for j=1:length(detectedBoxes(:,1))
            rectangle('Position',[detectedBoxes(j,1) detectedBoxes(j,2) detectedBoxes(j,3)-detectedBoxes(j,1) detectedBoxes(j,4)-detectedBoxes(j,2)], 'LineWidth',2, 'EdgeColor','r');
        end
       %}
        
        overlap = zeros (size(detectedBoxes,1),1);
        lenoverlap=length(overlap);
        for o=1:lenoverlap
            overlap(o) = calculateOverlap(im,detectedBoxes(o,1), detectedBoxes(o,2),detectedBoxes(o,3), detectedBoxes(o,4),...
                pos(i).x1, pos(i).y1, pos(i).x2, pos(i).y2);
            %if mod(o,10)==0
                 %fprintf('calculating overlap: %d/%d\n', o, lenoverlap);
            %end
        end
        detectedBoxes(find(overlap <=min_overlap),:)=[];
        %scores = sort(detectedBoxes(:,end),'descend');
        %for k=1:length(scores)
        %    sortedBoxes(k,:) = detectedBoxes(find(detectedBoxes(:,end)==scores(k)));
        %end
        %find the best scoring one
        bestBox = detectedBoxes(find(detectedBoxes(:,end)==max(detectedBoxes(:,end))),1:4);
        %imshow(im);
        %drawRectangle(bestBox, 'r');
        %drawRectangle([pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2], 'b');
        if (length(bestBox)~=0)
            boxes(i).im = pos(i).im;
            boxes(i).x1 = bestBox(1);
            boxes(i).y1 = bestBox(2);
            boxes(i).x2 = bestBox(3);
            boxes(i).y2 = bestBox(4);
        else 
            boxes(i).im = pos(i).im;
            boxes(i).x1 = pos(i).x1;
            boxes(i).y1 = pos(i).y1;
            boxes(i).x2 = pos(i).x2;
            boxes(i).y2 = pos(i).y2;           
        end
        save([cachedir name '_train_latentPosExamples'], 'boxes');
    end
    %save([cachedir cls '_train'], 'pos_latentRoot');
end

function drawRectangle(box, color)

rectangle('Position',[box(1,1) box(1,2) box(1,3)-box(1,1) box(1,4)-box(1,2)], ...
    'LineWidth',2, 'EdgeColor',color);
end


function overlap=calculateOverlap(im,x1a,y1a,x2a,y2a, x1b,y1b,x2b,y2b)
        
      %Make sure coordinates are withing image
%tic

           imMask1 = zeros(size(im(:,:,1)));
           imMask2 = zeros(size(im(:,:,1)));  
       
      %window = subarray(im, y1, y2, x1, x2, 1);
     
      x_img_end = size(im(:,:,1),2);
      y_img_end = size(im(:,:,1),1);
      
      
      y1a=floor(y1a);
      y2a=floor(y2a);
      x1a=floor(x1a);
      x2a=floor(x2a);
      
      y1b=ceil(y1b);
      y2b=ceil(y2b);
      x1b=ceil(x1b);
      x2b=ceil(x2b);
      
      if (y1a < 1) 
          y1a=1;
      end
      if (y2a > y_img_end) 
          y2a=y_img_end;
      end
      if (x1a < 1) 
          x1a=1;
      end
      if (x2a > x_img_end) 
          x2a=x_img_end;
      end
      if (y1b < 1) 
          y1b=1;
      end
      if (y2b > y_img_end) 
          y2b=y_img_end;
      end
      if (x1b < 1) 
          x1b=1;
      end
      if (x2b > x_img_end) 
          x2b=x_img_end;
      end
      
      
      imMask1(y1a:y2a, x1a:x2a)=1;
      imMask2(y1b:y2b, x1b:x2b)=1;
      
      %Pad rectangles to make sure they're the same size
    %{  
      if (size(imMask1,1)>size(imMask2,1))
          newRow = zeros(size(imMask1,1)-size(imMask2,1),size(imMask1,2));
          imMask2 = [imMask2;newRow];
      elseif (size(imMask2,1)>size(imMask1,1))
          newRow = zeros(size(imMask2,1)-size(imMask1,1),size(imMask2,2));
          imMask1 = [imMask1;newRow];
      end
      if (size(imMask1,2)>size(imMask2,2))
          newCol = zeros(size(imMask1,2),size(imMask1,2)-size(imMask2,2));
          imMask2 = [imMask2,newCol];
      elseif (size(imMask2,2)>size(imMask1,2))
          newCol = zeros(size(imMask2,2),size(imMask1,2)-size(imMask1,2));
          imMask1 = [imMask1,newCol];
      end
     %}  
      
      %Union
      imMask_union = imMask1 + imMask2;
      
      intersection_size = length(imMask_union(find(imMask_union==2)));
      union_size = length(imMask_union(find(imMask_union==1)))+0.5*intersection_size;
      overlap = intersection_size/union_size;
  %toc
end
