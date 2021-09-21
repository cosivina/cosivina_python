'''
Script for running scene representation simulator.

The scripts loads the architecture from launcherSceneRepresentation (in
the original cosivina examples) and runs it with timing. Then it loads
a variant using the FFT method for all 2D convolutions (generated in
the matching Matlab script), und runs the simulation again to compare
performance.
'''

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


## load and run original version of scene representation simulator

# load settings (preset from cosivina examples folder)
sim = Simulator(file = 'presetSceneRepresentation.json')

# activate stimuli
sim.setElementParameters(['i1 for vis_f1', 'i1 for vis_f2', 'i2 for vis_f1',
        'i2 for vis_f2', 'i3 for vis_f1', 'i3 for vis_f2'],
        ['amplitude'] * 6, [6] * 6)

# turn off noise and add a small spatial attention input to select one item (so we can compare Matlab and Python results)
sim.setElementParameters(['noise kernel atn_sr', 'noise kernel atn_sa',
        'i1 for atn_sr', 'i2 for atn_sr'], ['amplitude'] * 4, [0, 0, 0.2, 0.1])


tMax = 200

print('Running original version of scene representation simulator (first time) ...')
t0 = time.time()
sim.run(tMax, True)
t1 = time.time()
print(f'Time taken: {round(t1-t0, ndigits=3)} seconds')

print('Running original version of scene representation simulator (second time) ...')
t0 = time.time()
sim.run(tMax, True)
t1 = time.time()
print(f'Time taken: {round(t1-t0, ndigits=3)} seconds')


## load and run original version of scene representation simulator

# load settings (created in the accompanying Matlab file by replacing
# all 2D kernels with KernelFFT
simFFT = Simulator(file = 'presetSceneRepresentationFFT.json')

print('Running FFT version of scene representation simulator (first time) ...')
t0 = time.time()
simFFT.run(tMax, True)
t1 = time.time()
print(f'Time taken: {round(t1-t0, ndigits=3)} seconds')

print('Running FFT version of scene representation simulator (second time) ...')
t0 = time.time()
simFFT.run(tMax, True)
t1 = time.time()
print(f'Time taken: {round(t1-t0, ndigits=3)} seconds')


## save some field activations to mat file to compare them in Matlab

sim.saveComponentsToMat('sceneRepresentationResults.mat')

# # code example for manually saving individual components (or other variables)
# savemat('sceneRepresentationPython.mat',
#         {'vis_f1': simFFT.getComponent('vis_f1', 'activation'),
#         'wm_c1': simFFT.getComponent('wm_c1', 'activation')})






