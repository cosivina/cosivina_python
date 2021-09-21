% Script for a neural field simulator with two 1D fields interacting with a
% single 2D field.
% 
% The scripts sets up an architecture with three neural fields defined over two
% different feature spaces (based on launcherCoupling from the original cosivina
% examples, but using KernelFFT for the 2D interactions).
% 
% The script then runs the simulator and plots the field activations.


%% setting up the simulator

% shared parameters
sizeSpt = 100;
sizeFtr = 100;

sigma_exc = 5;
sigma_inh = 10;

% create simulator
sim = Simulator();

% create inputs
sim.addElement(GaussStimulus2D('stimulus v1', [sizeFtr, sizeSpt], sigma_exc, sigma_exc, 0, ...
  round(1/4*sizeFtr), round(1/4*sizeSpt), true, true));
sim.addElement(GaussStimulus2D('stimulus v2', [sizeFtr, sizeSpt], sigma_exc, sigma_exc, 0, ...
  round(3/4*sizeFtr), round(3/4*sizeSpt), true, true));

sim.addElement(GaussStimulus1D('stimulus s1', [1, sizeSpt], sigma_exc, 0, round(1/4*sizeSpt), true));
sim.addElement(GaussStimulus1D('stimulus s2', [1, sizeSpt], sigma_exc, 0, round(3/4*sizeSpt), true));
sim.addElement(SumInputs('stimulus sum s', [1, sizeSpt]), {'stimulus s1', 'stimulus s2'});

sim.addElement(GaussStimulus1D('stimulus f1', [1, sizeFtr], sigma_exc, 0, round(1/4*sizeFtr), true));
sim.addElement(GaussStimulus1D('stimulus f2', [1, sizeFtr], sigma_exc, 0, round(3/4*sizeFtr), true));
sim.addElement(SumInputs('stimulus sum f', [1, sizeSpt]), {'stimulus f1', 'stimulus f2'});

% create neural fields
sim.addElement(NeuralField('field v', [sizeFtr, sizeSpt], 20, -5, 4), {'stimulus v1', 'stimulus v2'});
sim.addElement(NeuralField('field s', [1, sizeSpt], 20, -5, 4), 'stimulus sum s');
sim.addElement(NeuralField('field f', [1, sizeFtr], 20, -5, 4), 'stimulus sum f');

% add lateral interactions
% sim.addElement(LateralInteractions2D('v -> v', [sizeFtr, sizeSpt], sigma_exc, sigma_exc, 0, sigma_inh, sigma_inh, 0, 0, ...
%   false, false, true), 'field v', 'output', 'field v');
sim.addElement(KernelFFT('v -> v', [sizeFtr, sizeSpt], [sigma_exc, sigma_exc], 0, [sigma_inh, sigma_inh], 0, 0, ...
  [true, true]), 'field v', 'output', 'field v');
sim.addElement(SumAllDimensions('sum v', [sizeFtr, sizeSpt]), 'field v', 'output');
sim.addElement(LateralInteractions1D('s -> s', [1, sizeSpt], sigma_exc, 0, sigma_inh, 0, 0, true), ...
  'field s', 'output', 'field s');
sim.addElement(LateralInteractions1D('f -> f', [1, sizeSpt], sigma_exc, 0, sigma_inh, 0, 0, true), ...
  'field f', 'output', 'field f');

% projections from field v to 1D fields (uses sum along one dimension)
sim.addElement(GaussKernel1D('v -> s', [1, sizeSpt], sigma_exc, 0, true), 'sum v', 'verticalSum', 'field s');
sim.addElement(GaussKernel1D('v -> f', [1, sizeFtr], sigma_exc, 0, true), 'sum v', 'horizontalSum', 'field f');

% projections from 1D fields to field v (requires transpose for vertical dimension)
sim.addElement(GaussKernel1D('s -> v', [1, sizeSpt], sigma_exc, 0, true), 'field s', 'output', 'field v');
sim.addElement(GaussKernel1D('f -> v', [1, sizeFtr], sigma_exc, 0, true), 'field f', 'output');
sim.addElement(Transpose('transpose f -> v', [sizeFtr, 1]), 'f -> v', 'output', 'field v');

% noise
sim.addElement(NormalNoise('noise s', [1, sizeSpt], 1), [], [], 'field s');
sim.addElement(NormalNoise('noise f', [1, sizeFtr], 1), [], [], 'field f');
sim.addElement(NormalNoise('noise v', [sizeFtr, sizeSpt], 1), [], [], 'field v');

sim.loadSettings('presetCouplingFFT.json');


%% run the simulation

tMax = 200;
fprintf('Running simulation ...\n');
tic;
sim.run(tMax, true);
t = toc;
fprintf('Time taken: %0.3f seconds\n', t);


%% plot field activations

p = 0.1;
w = 0.55;
h = 0.15;

figure('Position', [400, 200, 500, 500]);
hAxSpt = axes('Units', 'normalized', 'Position', [2*p+h, p, w, h], 'XLim', [1, sizeSpt], 'YLim', [-10, 10], ...
    'nextPlot', 'add');
xlabel('location'); ylabel('activation');
hAxFtr = axes('Units', 'normalized', 'Position', [p, 2*p+h, h, w], 'YLim', [1, sizeFtr], 'XLim', [-10, 10], ...
    'XDir', 'reverse', 'YAxisLocation', 'right', 'nextPlot', 'add');
xlabel('activation'); ylabel('feature');
hAxCnj = axes('Units', 'normalized', 'Position', [2*p+h, 2*p+h, w, w]);
ylabel('feature');

plot(1:sizeSpt, zeros(1, sizeSpt), '--k', 1:sizeSpt, sim.getComponent('field s', 'activation'), 'b', 'Parent', hAxSpt)
plot(zeros(1, sizeFtr), 1:sizeFtr, '--k', sim.getComponent('field f', 'activation'), 1:sizeFtr, 'b', 'Parent', hAxFtr)
imagesc(sim.getComponent('field v', 'activation'), [-10, 10])
set(hAxCnj, 'YAxisLocation', 'right', 'YDir', 'normal')
colormap(jet)

