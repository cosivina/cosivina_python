% Script for running scene representation simulator.
% 
% The scripts loads the architecture from launcherSceneRepresentation (in the
% original cosivina examples) and runs it with timing. Then it converts the
% simulator to use the FFT method for all 2D convolutions, und runs the
% simulation again.

fprintf('Loading scene representation simulator ...\n');

% load settings (preset from cosivina examples folder)
sim = Simulator('file', 'presetSceneRepresentation.json');

% activate stimuli
sim.setElementParameters({'i1 for vis_f1', 'i1 for vis_f2', 'i2 for vis_f1', 'i2 for vis_f2', 'i3 for vis_f1', 'i3 for vis_f2'}, ...
    repmat({'amplitude'}, [1, 6]), repmat(6, [1, 6]));

% turn off noise and add a small spatial attention input to select one item (so we can compare Matlab and Python results)
sim.setElementParameters({'noise kernel atn_sr', 'noise kernel atn_sa', 'i1 for atn_sr', 'i2 for atn_sr'}, ...
    repmat({'amplitude'}, [1, 4]), [0, 0, 0.2, 0.1]);

% run original version
fprintf('Running original version of scene representation simulator ...\n');
tMax = 200;
tic;
sim.run(tMax, true);
t = toc;
fprintf('Time taken: %f seconds\n', t);

% convert all 2D interactions to FFT elements (some are already using FFT kernels in the original)
simFFT = switchToFFT(sim, false, false, 0, 0, ...
    {'atn_c1 -> atn_c1', 'wm_c1 -> atn_c1', 'atn_c1 -> wm_c1', 'atn_c2 -> atn_c2', 'wm_c2 -> atn_c2', 'atn_c2 -> wm_c2'});

% save this version as json file so we can load in python
simFFT.saveSettings('presetSceneRepresentationFFT.json');

% run FFT version
fprintf('Running FFT version of scene representation simulator ...\n');
tic;
simFFT.run(tMax, true);
t = toc;
fprintf('Time taken: %f seconds\n', t);


%% compare field activations between Matlab and Python versions

% read saved components from python simulation and extract element labels
pyResults = load('sceneRepresentationResults.mat');
pyLabels = cell(numel(pyResults.elements), 1);
for i = 1 : numel(pyResults.elements)
    pyLabels{i} = pyResults.elements{i}.label;
end

% determine simulator elements of class NeuralField
neuralFieldLabels = {};
for i = 1 : sim.nElements
    if isa(sim.elements{i}, 'NeuralField')
        neuralFieldLabels{end+1} = sim.elements{i}.label; %#ok<SAGROW>
    end
end

% compare field activations
maxDiff = 0;
for i = 1 : numel(neuralFieldLabels)
    el = sim.getElement(neuralFieldLabels{i});
    s = pyResults.elements{strcmp(neuralFieldLabels{i}, pyLabels)};
    
    aMatlab = el.activation;
    aPython = s.activation;
    maxDiff = max(maxDiff, max(abs(aMatlab(:) - aPython(:))));
end

fprintf('Maximum difference in field or node activation between Matlab and Python versions: %e\n', maxDiff);

