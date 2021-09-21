'''
Script for a neural field simulator with two 1D fields interacting with
a single 2D field.

The scripts sets up an architecture with three neural fields defined
over two different feature spaces (based on launcherCoupling from the
original cosivina examples, but using KernelFFT for the 2D
interactions).

The script then runs the simulator and plots the field activations.
'''


import time
import numpy as np
import matplotlib.pyplot as plt


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


## create simulation

# Note: We could create the simulator by just loading from the json file,
# sim = Simulator(file = 'presetCouplingFFT.json')
# but we want to show the code for creating a simulation from scratch

print('Creating simulator object...')

# shared parameters
sizeFtr = 100
sizeSpt = 100
sigma_exc = 5
sigma_inh = 10

# create simulator object
sim = Simulator()

# create inputs
sim.addElement(GaussStimulus2D('stimulus v1', (sizeFtr, sizeSpt), sigma_exc, sigma_exc, 0,
        round(1/4*sizeFtr), round(1/4*sizeSpt), 0, 1))
sim.addElement(GaussStimulus2D('stimulus v2', (sizeFtr, sizeSpt), sigma_exc, sigma_exc, 0,
        round(3/4*sizeFtr), round(3/4*sizeSpt), 0, 1))

sim.addElement(GaussStimulus1D('stimulus s1', (1, sizeSpt), sigma_exc, 0, round(1/4*sizeSpt), True))
sim.addElement(GaussStimulus1D('stimulus s2', (1, sizeSpt), sigma_exc, 0, round(3/4*sizeSpt), True))
sim.addElement(SumInputs('stimulus sum s', (1, sizeSpt)), ['stimulus s1', 'stimulus s2'])

sim.addElement(GaussStimulus1D('stimulus f1', (1, sizeFtr), sigma_exc, 0, round(1/4*sizeFtr), True))
sim.addElement(GaussStimulus1D('stimulus f2', (1, sizeFtr), sigma_exc, 0, round(3/4*sizeFtr), True))
sim.addElement(SumInputs('stimulus sum f', (1, sizeSpt)), ['stimulus f1', 'stimulus f2'])

# create neural fields
sim.addElement(NeuralField('field v', (sizeFtr, sizeSpt), 20, -5, 4), ['stimulus v1', 'stimulus v2'])
sim.addElement(NeuralField('field s', (1, sizeSpt), 20, -5, 4), 'stimulus sum s')
sim.addElement(NeuralField('field f', (1, sizeFtr), 20, -5, 4), 'stimulus sum f')

# add lateral interactions
sim.addElement(KernelFFT('v -> v', (sizeFtr, sizeSpt), [sigma_exc, sigma_exc], 0,
        [sigma_inh, sigma_inh], 0, 0, [True, True]), 'field v', 'output', 'field v')
sim.addElement(SumAllDimensions('sum v', (sizeFtr, sizeSpt)), 'field v', 'output')
sim.addElement(LateralInteractions1D('s -> s', (1, sizeSpt), sigma_exc, 0, sigma_inh, 0, 0, True), 'field s', 'output', 'field s')
sim.addElement(LateralInteractions1D('f -> f', (1, sizeSpt), sigma_exc, 0, sigma_inh, 0, 0, True),
  'field f', 'output', 'field f')

# projections from field v to 1D fields (uses sum along one dimension)
sim.addElement(GaussKernel1D('v -> s', (1, sizeSpt), sigma_exc, 0, True), 'sum v', 'verticalSum', 'field s')
sim.addElement(GaussKernel1D('v -> f', (1, sizeFtr), sigma_exc, 0, True), 'sum v', 'horizontalSum', 'field f')

# projections from 1D fields to field v (requires transpose for vertical dimension)
sim.addElement(GaussKernel1D('s -> v', (1, sizeSpt), sigma_exc, 0, True), 'field s', 'output', 'field v')
sim.addElement(GaussKernel1D('f -> v', (1, sizeFtr), sigma_exc, 0, True), 'field f', 'output')
sim.addElement(Transpose('transpose f -> v', (sizeFtr, 1)), 'f -> v', 'output', 'field v')

# noise
sim.addElement(NormalNoise('noise s', (1, sizeSpt), 1), [], [], 'field s')
sim.addElement(NormalNoise('noise f', (1, sizeFtr), 1), [], [], 'field f')
sim.addElement(NormalNoise('noise v', (sizeFtr, sizeSpt), 1), [], [], 'field v')

# load settings from file
sim.loadSettings('presetCouplingFFT.json')


## run

tMax = 200

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


## plot resulting activation

plt.ion()

p = 0.1
w = 0.55
h = 0.15

# setting up the axes
hFig = plt.figure(figsize=[5, 5])
hAxSpt = plt.axes([2*p+h, p, w, h], xlim=[1, sizeSpt], ylim=[-10, 10],
    xlabel='location', ylabel='activation')
hAxFtr = plt.axes([p, 2*p+h, h, w], ylim=[1, sizeFtr], xlim=[10, -10],
    ylabel='feature', xlabel='activation')
hAxFtr.yaxis.tick_right()
hAxCnj = plt.axes([2*p+h, 2*p+h, w, w])
hAxCnj.yaxis.tick_right()

# plotting field activations
# Note: All components are 2D numpy arrays, to plot 1D activations you need
# to select component[0]
hAxSpt.plot(np.arange(1, sizeSpt+1), np.zeros(sizeSpt), '--k',
    np.arange(1, sizeSpt+1), sim.getComponent('field s', 'activation')[0], 'b')
hAxFtr.plot(np.zeros(sizeSpt), np.arange(1, sizeSpt+1), '--k',
    sim.getComponent('field f', 'activation')[0], np.arange(1, sizeFtr+1), 'b')
hAxCnj.imshow(sim.getComponent('field v', 'activation'), vmin=-10, vmax=10,
    cmap=plt.get_cmap('jet'), origin='lower')



