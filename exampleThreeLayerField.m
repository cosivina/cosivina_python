% Script for a three-layer neural field simulator.
% 
% The scripts sets up an architecture with three one-dimensional neural fields:
% two excitatory fields, u and w, and a shared inhibitory field, v. Fields u and
% w feature local self-excitation and project to each other and to field v in a
% local excitatory fashion. Field v inhibits both excitatory fields, either
% locally or globally.
% 
% The script then runs the simulator and measures run time.


%% setting up the simulator

% shared parameters
fieldSize = 180;
sigma_exc = 5;
sigma_inh = 10;

% create simulator object
sim = Simulator();

% create inputs (and sum for visualization)
sim.addElement(GaussStimulus1D('stimulus 1', fieldSize, sigma_exc, 0, round(1/4*fieldSize), true, false));
sim.addElement(GaussStimulus1D('stimulus 2', fieldSize, sigma_exc, 0, round(1/2*fieldSize), true, false));
sim.addElement(GaussStimulus1D('stimulus 3', fieldSize, sigma_exc, 0, round(3/4*fieldSize), true, false));
sim.addElement(SumInputs('stimulus sum', fieldSize), {'stimulus 1', 'stimulus 2', 'stimulus 3'});
sim.addElement(ScaleInput('stimulus scale w', fieldSize, 0), 'stimulus sum');

% create neural field
sim.addElement(NeuralField('field u', fieldSize, 20, -5, 4), 'stimulus sum');
sim.addElement(NeuralField('field v', fieldSize, 5, -5, 4));
sim.addElement(NeuralField('field w', fieldSize, 20, -5, 4), 'stimulus scale w');

% shifted input sum (for plot) - removed because not currently supported in python
% sim.addElement(SumInputs('shifted stimulus sum', fieldSize), {'stimulus sum', 'field u'}, {'output', 'h'});
% sim.addElement(SumInputs('shifted stimulus sum w', fieldSize), {'stimulus scale w', 'field w'}, {'output', 'h'});

% create interactions
sim.addElement(GaussKernel1D('u -> u', fieldSize, sigma_exc, 0, true, true), 'field u', 'output', 'field u');
sim.addElement(GaussKernel1D('u -> v', fieldSize, sigma_exc, 0, true, true), 'field u', 'output', 'field v');
sim.addElement(GaussKernel1D('u -> w', fieldSize, sigma_exc, 0, true, true), 'field u', 'output', 'field w');

sim.addElement(GaussKernel1D('v -> u (local)', fieldSize, sigma_inh, 0, true, true), 'field v', 'output', 'field u');
sim.addElement(GaussKernel1D('v -> w (local)', fieldSize, sigma_inh, 0, true, true), 'field v', 'output', 'field w');
sim.addElement(SumDimension('sum v', 2, 1, 1), 'field v', 'output');
sim.addElement(ScaleInput('v -> u (global)', fieldSize, 0), 'sum v', 'output', 'field u');
sim.addElement(ScaleInput('v -> w (global)', fieldSize, 0), 'sum v', 'output', 'field w');

sim.addElement(GaussKernel1D('w -> u', fieldSize, sigma_exc, 0, true, true), 'field w', 'output', 'field u');
sim.addElement(GaussKernel1D('w -> v', fieldSize, sigma_exc, 0, true, true), 'field w', 'output', 'field v');
sim.addElement(GaussKernel1D('w -> w', fieldSize, sigma_exc, 0, true, true), 'field w', 'output', 'field w');

% create noise stimulus and noise kernel
sim.addElement(NormalNoise('noise u', fieldSize, 1));
sim.addElement(GaussKernel1D('noise kernel u', fieldSize, 0, 2, true, true), 'noise u', 'output', 'field u');
sim.addElement(NormalNoise('noise v', fieldSize, 1));
sim.addElement(GaussKernel1D('noise kernel v', fieldSize, 0, 1, true, true), 'noise v', 'output', 'field v');
sim.addElement(NormalNoise('noise w', fieldSize, 1));
sim.addElement(GaussKernel1D('noise kernel w', fieldSize, 0, 2, true, true), 'noise w', 'output', 'field w');


%% run the simulator in the GUI

tMax = 1000;

tic;
sim.run(tMax, true);
t = toc;
fprintf('Time taken: %f seconds\n', t);


