'''
Script for a three-layer neural field simulator.

The scripts sets up an architecture with three one-dimensional neural
fields: two excitatory fields, u and w, and a shared inhibitory field, v.
Fields u and w feature local self-excitation and project to each other
and to field v in a local excitatory fashion. Field v inhibits both
excitatory fields, either locally or globally.

The script then runs the simulator, first with timing, then with online
plotting.
'''

import numpy as np
import matplotlib.pyplot as plt
import time


### choose here whether cosivina is used with or without numba ###

from cosivina.nonumba import *
# from cosivina.numba import *

if options.useNumba:
    print('Using cosvina with numba (just-in-time compilation). Each function '
            'will be compiled when first called, which can take a long time, '
            'but subsequent function calls will be substantially faster.\n')
else:
    print('Using cosvina without numba.\n')

print('Note: Changing between numba and no-numba mode requires restart of the kernel,\n')


## create sim

# Note: We could create the simulator by just loading from the json file:
# sim = Simulator(file = 'presetThreeLayerField_changeDetection.json')

print('Creating simulator object...')

# shared parameters
fieldSize = (1, 180) # important: size for elements must be defined as a tuple
sigma_exc = 5
sigma_inh = 10

# create simulator object
sim = Simulator()

# create inputs (and sum for visualization)
sim.addElement(GaussStimulus1D('stimulus 1', fieldSize, sigma_exc, 0, round(1/4*fieldSize[1]), True, False))
sim.addElement(GaussStimulus1D('stimulus 2', fieldSize, sigma_exc, 0, round(1/2*fieldSize[1]), True, False))
sim.addElement(GaussStimulus1D('stimulus 3', fieldSize, sigma_exc, 0, round(3/4*fieldSize[1]), True, False))
sim.addElement(SumInputs('stimulus sum', fieldSize), ['stimulus 1', 'stimulus 2', 'stimulus 3'])
sim.addElement(ScaleInput('stimulus scale w', fieldSize, 0), 'stimulus sum')

# create neural field
sim.addElement(NeuralField('field u', fieldSize, 20, -5, 4), 'stimulus sum')
sim.addElement(NeuralField('field v', fieldSize, 5, -5, 4))
sim.addElement(NeuralField('field w', fieldSize, 20, -5, 4), 'stimulus scale w')

# # shifted input sum (for plot) - currently not supported in numba mode (h is not a component of NeuralField)
# sim.addElement(SumInputs('shifted stimulus sum', fieldSize), ['stimulus sum', 'field u'], ['output', 'h'])
# sim.addElement(SumInputs('shifted stimulus sum w', fieldSize), ['stimulus scale w', 'field w'], ['output', 'h'])

# create interactions
sim.addElement(GaussKernel1D('u -> u', fieldSize, sigma_exc, 0, True, True), 'field u', 'output', 'field u')
sim.addElement(GaussKernel1D('u -> v', fieldSize, sigma_exc, 0, True, True), 'field u', 'output', 'field v')
sim.addElement(GaussKernel1D('u -> w', fieldSize, sigma_exc, 0, True, True), 'field u', 'output', 'field w')

sim.addElement(GaussKernel1D('v -> u (local)', fieldSize, sigma_inh, 0, True, True), 'field v', 'output', 'field u')
sim.addElement(GaussKernel1D('v -> w (local)', fieldSize, sigma_inh, 0, True, True), 'field v', 'output', 'field w')

# the dimension(s) to sum over must be specified as a numpy array
sim.addElement(SumDimension('sum v', np.array([2]), (1, 1), 1), 'field v', 'output')
sim.addElement(ScaleInput('v -> u (global)', fieldSize, 0), 'sum v', 'output', 'field u')
sim.addElement(ScaleInput('v -> w (global)', fieldSize, 0), 'sum v', 'output', 'field w')

sim.addElement(GaussKernel1D('w -> u', fieldSize, sigma_exc, 0, True, True), 'field w', 'output', 'field u')
sim.addElement(GaussKernel1D('w -> v', fieldSize, sigma_exc, 0, True, True), 'field w', 'output', 'field v')
sim.addElement(GaussKernel1D('w -> w', fieldSize, sigma_exc, 0, True, True), 'field w', 'output', 'field w')

# create noise stimulus and noise kernel
sim.addElement(NormalNoise('noise u', fieldSize, 1));
sim.addElement(GaussKernel1D('noise kernel u', fieldSize, 0, 2, True, True), 'noise u', 'output', 'field u')
sim.addElement(NormalNoise('noise v', fieldSize, 1))
sim.addElement(GaussKernel1D('noise kernel v', fieldSize, 0, 1, True, True), 'noise v', 'output', 'field v')
sim.addElement(NormalNoise('noise w', fieldSize, 1))
sim.addElement(GaussKernel1D('noise kernel w', fieldSize, 0, 2, True, True), 'noise w', 'output', 'field w')

# load parameter settings
sim.loadSettings('presetThreeLayerField_changeDetection.json')


## running the simulation

tMax = 1000

print('Running simulation (first time) ...')
t0 = time.time()
sim.run(tMax, True)
t1 = time.time()
print(f'Time taken: {round(t1-t0, ndigits=3)} seconds')

print('Running simulation (second time) ...')
t0 = time.time()
sim.run(tMax, True)
t1 = time.time()
print(f'Time taken: {round(t1-t0, ndigits=3)} seconds')


## online plotting of field activations

print('\nRunning simulation with online plotting (plotting may be slow in some environments) ...')

# turn on interactive plotting
plt.ion()

# re-initialize
sim.init()

# prepare axes
fig, axes = plt.subplots(3, 1)
for i in range(3):
    axes[i].set_xlim(0, fieldSize[1])
    axes[i].set_ylim(-15, 15)
    axes[i].set_ylabel('activation')
axes[2].set_xlabel('feature space')

# plot initial state
x = np.arange(fieldSize[1])
plot_stim, plot_u = axes[0].plot(x, sim.getComponent('stimulus sum', 'output')[0],
        x, sim.getComponent('field u', 'activation')[0], color='b')
plot_stim.set_color('g')
plot_v, = axes[1].plot(x, sim.getComponent('field v', 'activation')[0], color='b')
plot_stimw, plot_w = axes[2].plot(x, sim.getComponent('stimulus scale w', 'output')[0],
        x, sim.getComponent('field w', 'activation')[0], color='b')
plot_stimw.set_color('g')

# run simulation
tMax = 100
for t in range(tMax):
    sim.step()

    plot_u.set_ydata(sim.getComponent('field u', 'activation')[0])
    plot_v.set_ydata(sim.getComponent('field v', 'activation')[0])
    plot_w.set_ydata(sim.getComponent('field w', 'activation')[0])
    plot_stim.set_ydata(sim.getComponent('stimulus sum', 'output')[0])
    plot_stimw.set_ydata(sim.getComponent('stimulus scale w', 'output')[0])
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)



