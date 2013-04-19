function model = pascal_train(cls, n)

% model = pascal_train(cls)
% Train a model using the PASCAL dataset.

globals; 


[pos, neg] = pascal_data(cls);

% train root filter using warped positives & random negatives
try
  load([cachedir cls '_randomRoot']);
catch
  model = initmodel(pos);
  model = train(cls, cachedir,model, pos, neg,0);
  save([cachedir cls '_randomRoot'], 'model');
end

% PUT YOUR CODE HERE
% TODO: Train the rest of the DPM (latent root position, part filters, ...)

% Train latent root position using positives that score best on root filter
% trained from before. 


try
  load([cachedir cls '_random_latentRoot']);
catch
  model = train(cls, cachedir, model, pos, neg, 1);
  save([cachedir cls '_random_latentRoot'], 'model');
end


% train a DPM part filters on random negatives

%try
%  load([cachedir cls '_randDPM']);
%catch
  numparts=6;
  model_dpm = initialize_parts(model,numparts);
  name = [class 'dpm'];
  model_dpm=train_dpm(name, cachedir, model_dpm, pos);
  %model_pdm = train_pdm(cls, model_pdm, pos, neg,0,1); %train_dpm
  %save([cachedir cls '_randDPM'], 'model');
%end
