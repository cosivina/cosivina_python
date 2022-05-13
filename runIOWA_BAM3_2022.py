'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COSIVINA IOWA Task Simulator
%
% John Spencer
% building on in-line simulator from Sebastian Schneegans
% and IOWA-C expansion from 2022 paper
%
% origin: stripped down version of the biased competition model (spatial pathway only)
% model components:
% field_a - spatial attention field
% field_s - saccade motor field
% neuron_r - saccade reset neuron
% neuron_x - fixation neuron, excites foveal position in field_a
% neuron_g - gaze-change neuron, inhibits foveal position in field_a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

'''%% constant identifiers %%'''
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
import numpy.matlib
from scipy.io import savemat
from scipy.ndimage.filters import convolve

# from conv2emulator import conv2
def conv2(x,y,mode='same'):
    """
    Emulate the function conv2 from Mathworks.

    Usage:

    z = conv2(x,y,mode='same')

    TODO: 
     - Support other modes than 'same' (see conv2.m)
    """

    if not(mode == 'same'):
        raise Exception("Mode not supported")

    # Add singleton dimensions
    if (len(x.shape) < len(y.shape)):
        dim = x.shape
        for i in range(len(x.shape),len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)
    elif (len(y.shape) < len(x.shape)):
        dim = y.shape
        for i in range(len(y.shape),len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = ()

    # Apparently, the origin must be set in a special way to reproduce
    # the results of scipy.signal.convolve and Matlab
    for i in range(len(x.shape)):
        if ( (x.shape[i] - y.shape[i]) % 2 == 0 and
             x.shape[i] > 1 and
             y.shape[i] > 1):
            origin = origin + (-1,)
        else:
            origin = origin + (0,)

    z = convolve(x,y, mode='constant', origin=origin)

    return z


### choose here whether cosivina is used with or without numba ###

# from cosivina.nonumba import *
from cosivina.numba import *

if options.useNumba:
    print('Using cosvina with numba (just-in-time compilation). Each function '
            'will be compiled when first called, which can take a long time, '
            'but subsequent function calls will be substantially faster.\n')
else:
    print('Using cosvina without numba.\n')

print('Note: Changing between numba and no-numba mode may require restarting the kernel.\n')

starttime = time.time()

'''% 0 for batch mode, 1 for auto mode with visualization, 2 for multicore (also need to switch to 'parfor' in line 193 below)'''
mode = 0


'''%% experiment settings %%'''
targetSize = 46 
targetEccentricity = 110
tCueStart = 400
tCueEnd = 500 
tTargetStart = tCueEnd + 100
minSaccLatency = 100
maxSaccLatency = 2600
tMax = tTargetStart + maxSaccLatency
tTargetEnd = tMax
tToneBoostStart = tCueStart
tToneBoostEnd = tCueEnd

breakOnFirstSaccade = True

'''%% model parameters %%

% Parameters are all defined in a struct p for easier management and
% storage. For each parameter, either a single value should be specified,
% or a vector of three values for ages [5MO, 7MO, 10MO]. Additional
% modifications on these parameters (for trying out different settings in a
% batch) can be specified below.
'''

cueSize = 17 
fixationSize = 27
spatialFieldSizeT = (1, 301) 
spatialFieldSize = 301 
spatialHalfSize = (spatialFieldSize-1)/2

tau_a = 60
tau_s = 60
h_a = -5 
h_s = -5 
h_r = -5 
h_x = -5 
h_g = -5 
beta_a = 1 
beta_s = 4 
beta_r = 4 
beta_x = 1 
beta_g = 1 
q_r = 0.05
q_x = 0.05
sigma_q = 2

'''% smoothing of visual input'''
c_v_exc = 8 
sigma_v_exc = 2.5
c_v_inh = 0 
sigma_v_inh = 5
c_v_gi = 0

'''% spatial attention field'''
sigma_aa_exc = 8 
sigma_aa_inh = 20 

'''% saccade motor field'''
sigma_ss_exc = 8
c_ss_inh = 0 
sigma_ss_inh = 8

sigma_sa = 10 
c_as = 0 
sigma_as = 10

c_rr = 2.5

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% NEW -- some details needed for 2022 paper'''
c_gg = 2.5 
c_xx = 5.0 
c_gx_inh = 1.0 

c_ax = 5 
sigma_ax_exc = 8
c_ag_inh = 10 
c_ag_exc = 2 
sigma_ag_exc = 8

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
c_xr_inh = 5 
c_gr_inh = 5
theta_saccStart = 0.95 
theta_saccEnd = 0.05 
c_sacc = 0.0015 
stimStrength_g_tone = 4 
stimStrength_g_base = 4
stimStrength_x = 7 
saccadeLatencyOffset = 75
kernelWidthMultiplier = 4


'''%% simulation settings %%'''
DOUBLE_CUE = 0
INVALID_CUE = 1
NO_CUE = 2
TONE_CUE = 3
VALID_CUE = 4
CDOUBLE_CUE = 5
CINVALID_CUE = 6
CNO_CUE = 7
CTONE_CUE = 8
CVALID_CUE = 9

if mode == 1:
    plotLatencies = False
    plotErrorRates = False
    printTrials = True
    saveResults =False
    storeHistory = True
    nRepeats = 1
    noise = True

    experi = 5

    if experi == 1: 
        INFANT_5MO = 0
        INFANT_7MO = 1
        INFANT_10MO = 2
        conditions = [DOUBLE_CUE, INVALID_CUE, NO_CUE, TONE_CUE, VALID_CUE]
        ages = [INFANT_5MO, INFANT_7MO, INFANT_10MO]
        switchcondition = 5

    if experi == 2:
        INFANT_5MO = 0
        INFANT_7MO = 1
        INFANT_10MO = 2
        conditions = [DOUBLE_CUE, INVALID_CUE, NO_CUE, TONE_CUE, VALID_CUE, CDOUBLE_CUE, CINVALID_CUE, CNO_CUE, CTONE_CUE, CVALID_CUE]
        ages = [INFANT_5MO, INFANT_7MO, INFANT_10MO]
        switchcondition = 5

    if experi == 3:
        stimStrength_g_tone = 0
        INFANT_6MO = 0
        INFANT_18MO = 1
        INFANT_30MO = 2
        conditions = [DOUBLE_CUE, INVALID_CUE, NO_CUE, VALID_CUE, CDOUBLE_CUE, CINVALID_CUE, CNO_CUE, CVALID_CUE]
        ages = [INFANT_6MO, INFANT_18MO, INFANT_30MO]
        switchcondition = 4

    if experi == 4: 
        stimStrength_g_tone = 0 
        INFANT_30MO = 0
        INFANT_42MO = 1
        INFANT_54MO = 2
        conditions = [DOUBLE_CUE, INVALID_CUE, NO_CUE, VALID_CUE, CDOUBLE_CUE, CINVALID_CUE, CNO_CUE, CVALID_CUE]
        ages = [INFANT_30MO, INFANT_42MO, INFANT_54MO]
        switchcondition = 4

    if experi == 5: 
        INFANT_5MO = 0
        INFANT_7MO = 1
        INFANT_10MO = 2
        conditions = [DOUBLE_CUE, INVALID_CUE]
        ages = [INFANT_10MO]
        switchcondition = 4

    N_CONDITIONS = np.size(conditions) 
    N_AGES = np.size(ages)
    
else:
    plotLatencies = True
    plotErrorRates = True
    printTrials = False
    saveResults =True
    storeHistory = False
    nRepeats = 400
    noise = True

    '''% select desired experiment to simulate'''
    experi = 2
    resultBaseFilename = 'Exp'+str(experi)+'_Sims2numba.mat'

    '''%IOWA 2015 paper'''
    if experi == 1: 
        INFANT_5MO = 0
        INFANT_7MO = 1
        INFANT_10MO = 2
        #resultBaseFilename = ['Exp' str(experi) '_Sims1_']
        conditions = [DOUBLE_CUE, INVALID_CUE, NO_CUE, TONE_CUE, VALID_CUE]
        ages = [INFANT_5MO, INFANT_7MO, INFANT_10MO]
        switchcondition = 5 
    
    '''%IOWA 2022 paper'''
    if experi == 2:
        INFANT_5MO = 0
        INFANT_7MO = 1
        INFANT_10MO = 2
        #resultBaseFilename = ['Exp' str(experi) '_Sims2_']
        conditions = [DOUBLE_CUE, INVALID_CUE, NO_CUE, TONE_CUE, VALID_CUE, CDOUBLE_CUE, CINVALID_CUE, CNO_CUE, CTONE_CUE, CVALID_CUE]
        ages = [INFANT_5MO, INFANT_7MO, INFANT_10MO]
        switchcondition = 5
        
    '''%longitudinal VWM cohort 1'''
    if experi == 3:
        stimStrength_g_tone = 0 
        INFANT_6MO = 0
        INFANT_18MO = 1
        INFANT_30MO = 2
        #resultBaseFilename = ['Exp' str(experi) '_Sims1_']
        conditions = [DOUBLE_CUE, INVALID_CUE, NO_CUE, VALID_CUE, CDOUBLE_CUE, CINVALID_CUE, CNO_CUE, CVALID_CUE]
        ages = [INFANT_6MO, INFANT_18MO, INFANT_30MO]
        switchcondition = 4 

    '''%longitudinal VWM cohort 2'''
    if experi == 4:
        stimStrength_g_tone = 0
        INFANT_30MO = 0
        INFANT_42MO = 1
        INFANT_54MO = 2
        #resultBaseFilename = ['Exp' str(experi) '_Sims1_']
        conditions = [DOUBLE_CUE, INVALID_CUE, NO_CUE, VALID_CUE, CDOUBLE_CUE, CINVALID_CUE, CNO_CUE, CVALID_CUE]
        ages = [INFANT_30MO, INFANT_42MO, INFANT_54MO]
        switchcondition = 4

    N_CONDITIONS = np.size(conditions) 
    N_AGES = np.size(ages)


'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% DEVELOPMENTAL parameters'''

'''% these two experiments have same ages and conditions, so goal is to
% optimise parameters across all conditions and ages'''
if experi == 1 or experi == 2 or experi == 5:
    '''%reduction in noise over development'''
    q_a = [0.55*2, 0.5*2, 0.45*2]
    q_s = [0.6, 0.55, 0.5]

    '''%increasing strength of excitation/inhibition over development (both within
    %fields and in connections)...'''
    c_aa_exc = [22, 27, 27]
    c_aa_inh = [20, 24, 30]
    c_aa_gi = [0.045*1.95, 0.1*1.95, 0.1*1.95]
    c_ss_exc = [30, 40, 50]
    c_ss_gi = [0.75, 1.0, 1.25]
    c_sa = [6.75, 7.4, 9.75]
    c_rs = [0.9, 1.2, 1.5]
    c_sr_gi = [18, 24, 30]
    c_ar_gi = [18, 24, 30]


'''% this experiment hasn't been modelled yet. Different ages and no tone.
% Youngest age group is 6MO -- in-between the 5 and 7MO above. Note: just
% copying same developmental parameters here'''
if experi == 3:
    q_a = [0.55*2, 0.5*2, 0.45*2]
    q_s = [0.6, 0.55, 0.5]
    c_aa_exc = [22, 27, 27] 
    c_aa_inh = [20, 24, 30]
    c_aa_gi = [0.045*1.95, 0.1*1.95, 0.1*1.95]
    c_ss_exc = [30, 40, 50]
    c_ss_gi = [0.75, 1.0, 1.25] 
    c_sa = [6.75, 7.4, 9.75]
    c_rs = [0.9, 1.2, 1.5]
    c_sr_gi = [18, 24, 30]
    c_ar_gi = [18, 24, 30]

'''% this experiment hasn't been modelled yet. Different ages and no tone.
% Youngest age group is 30MO which is same as the oldest age in experi = 3. 
% Note: just copying same developmental parameters here.'''
if experi == 4:
    q_a = [0.55*2, 0.5*2, 0.45*2] 
    q_s = [0.6, 0.55, 0.5] 
    c_aa_exc = [22, 27, 27] 
    c_aa_inh = [20, 24, 30]
    c_aa_gi = [0.045*1.95, 0.1*1.95, 0.1*1.95] 
    c_ss_exc = [30, 40, 50]
    c_ss_gi = [0.75, 1.0, 1.25] 
    c_sa = [6.75, 7.4, 9.75]
    c_rs = [0.9, 1.2, 1.5]
    c_sr_gi = [18, 24, 30]
    c_ar_gi = [18, 24, 30]


'''%% prepare results struct and file %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

nConditions = np.size(conditions)
nAges = np.size(ages)

'''%% simulation %%
%%%%%%%%%%%%%%%%'''
latenciesCorrect = np.empty((N_CONDITIONS, N_AGES))
ratesCorrect = np.empty((N_CONDITIONS, N_AGES))
saccStartTimes = np.empty((nAges, N_CONDITIONS, nRepeats))
saccEndTimes = np.empty((nAges, N_CONDITIONS, nRepeats))
saccMetrics = np.empty((nAges, N_CONDITIONS, nRepeats))

for a in range(nAges):
        
    # %% create simulator object
    sim = Simulator()
   
    # %% create neural field
    sim.addElement(NeuralField('field a', spatialFieldSizeT, tau_a, h_a, beta_a))
    sim.addElement(NeuralField('field s', spatialFieldSizeT, tau_s, h_s, beta_s))
    sim.addElement(NeuralField('r', (1, 1), tau_s, h_r, beta_r))
    sim.addElement(NeuralField('x', (1, 1), tau_a, h_x, beta_x))
    sim.addElement(NeuralField('g', (1, 1), tau_a, h_g, beta_g))
   
    # %% create lateral interactions
    sim.addElement(LateralInteractions1D('input_aa', spatialFieldSizeT, sigma_aa_exc, c_aa_exc[a], sigma_aa_inh, c_aa_inh[a], -c_aa_gi[a], False, True, kernelWidthMultiplier), \
        'field a', 'output', 'field a')
    sim.addElement(LateralInteractions1D('input_ss', spatialFieldSizeT, sigma_ss_exc, c_ss_exc[a], sigma_ss_inh, c_ss_inh, -c_ss_gi[a], False, True, kernelWidthMultiplier), \
        'field s', 'output', 'field s')
    sim.addElement(ScaleInput('input_rr', (1, 1), c_rr), 'r', 'output', 'r')
    sim.addElement(ScaleInput('input_xx', (1, 1), c_xx), 'x', 'output', 'x')
    sim.addElement(ScaleInput('input_gg', (1, 1), c_gg), 'g', 'output', 'g')
   
    # %% create inputs and projections
   
    # %into field a
    sim.addElement(CustomStimulus('input_v', np.zeros(spatialFieldSizeT)), [], [], 'field a') 
    sim.addElement(GaussKernel1D('input_as', spatialFieldSizeT, sigma_as, c_as, False, True, kernelWidthMultiplier), 'field s', 'output', 'field a')
    sim.addElement(GaussStimulus1D('w_ax', spatialFieldSizeT, sigma_ax_exc, c_ax, spatialHalfSize, False, False))
    sim.addElement(PointwiseProduct('input_ax', spatialFieldSizeT), ['w_ax', 'x'], ['output', 'output'], 'field a')
    sim.addElement(GaussStimulus1D('w_ag_init', spatialFieldSizeT, sigma_ag_exc, -c_ag_inh, spatialHalfSize, False, False))
    sim.addElement(BoostStimulus('w_ag_offset', c_ag_exc)) 
    sim.addElement(SumInputs('w_ag', spatialFieldSizeT), ['w_ag_init', 'w_ag_offset'], [], [])
    sim.addElement(PointwiseProduct('input_ag', spatialFieldSizeT), ['w_ag', 'g'], ['output', 'output'], 'field a')
    sim.addElement(ScaleInput('input_ar', (1, 1), -c_ar_gi[a]), 'r', 'output', 'field a')
   
    # %into field s
    b1=np.array(range(int(-spatialHalfSize-1),int(spatialHalfSize)))
    suppressionPattern = 1 - np.exp(-0.5 * ((b1)-0)**2 / sigma_sa**2)
    sim.addElement(CustomStimulus('fovPattern', suppressionPattern))
    sim.addElement(PointwiseProduct('fovSup', spatialFieldSizeT), ['fovPattern', 'field a'], ['output', 'output'])
    sim.addElement(GaussKernel1D('input_sa', spatialFieldSizeT, sigma_sa, c_sa[a], False, True, kernelWidthMultiplier), 'fovSup', 'output', 'field s')
    sim.addElement(ScaleInput('input_sr', (1, 1), -c_sr_gi[a]), 'r', 'output', 'field s')
   
    # %into r
    # sim.addElement(SumDimension('field s -> node',  np.array([2]) , (1, 1)), 'field s', 'output')
    sim.addElement(SumAllDimensions('field s -> node', spatialFieldSizeT), 'field s', 'output')
    sim.addElement(ScaleInput('input_rs', (1, 1), c_rs[a]), 'field s -> node', 'fullSum', 'r')
   
    # %into x
    sim.addElement(BoostStimulus('input_x', 1.0), [], [], 'x') 
    sim.addElement(ScaleInput('input_xg', (1, 1), -c_gx_inh), 'g', 'output', 'x')
    sim.addElement(ScaleInput('input_xr', (1, 1), -c_xr_inh), 'r', 'output', 'x')
   
    # %into g
    sim.addElement(BoostStimulus('input_g', 1.0), [], [], 'g') 
    sim.addElement(ScaleInput('input_gx', (1, 1), -c_gx_inh), 'x', 'output', 'g')
    sim.addElement(ScaleInput('input_gr', (1, 1), -c_gr_inh), 'r', 'output', 'g')
   
    # %% create noise stimulus and noise kernel
    sim.addElement(NormalNoise('noise a', spatialFieldSizeT, 1)) 
    sim.addElement(GaussKernel1D('noise kernel a', spatialFieldSizeT, sigma_q, q_a[a], False, True, kernelWidthMultiplier), 'noise a', 'output', 'field a')
    sim.addElement(NormalNoise('noise s', spatialFieldSizeT, 1)) 
    sim.addElement(GaussKernel1D('noise kernel s', spatialFieldSizeT, sigma_q, q_s[a], False, True, kernelWidthMultiplier), 'noise s', 'output', 'field s')
    sim.addElement(NormalNoise('noise r', (1, 1), q_r), [], [], 'r')
    sim.addElement(NormalNoise('noise x', (1, 1), q_x), [], [], 'x')
    sim.addElement(NormalNoise('noise g', (1, 1), q_x), [], [], 'g')        
    
    
    # initial simulation set up
    sim.init()

    gui_speed = 5 

    '''%kernel used to create visual input'''
    kSize_v = min(round(kernelWidthMultiplier * max(sigma_v_exc, sigma_v_inh)), spatialFieldSize)
    b=np.array(range(-kSize_v-1,kSize_v))
    g1 = np.exp(-0.5 * ((b)-0)**2 / sigma_v_exc**2)
    g1 = g1 / np.sum(g1)
    g2 = np.exp(-0.5 * ((b)-0)**2 / sigma_v_inh**2)
    g2 = g2 / np.sum(g2)
    kernel_v = c_v_exc * g1 - c_v_inh * g2    
    
    '''%% simulation %%
    %%%%%%%%%%%%%%%%'''
    
    for j in range(nConditions):
        c = conditions[j]
        
        print('Condition '+str(c))
        
        '''%set up IOWA vs IOWA-C contrast'''
        if (c < switchcondition): 
            tFixEnd = 300
        else: 
            tFixEnd = 2600
        
        '''% all stimuli have a size of 3*spatialFieldSize (to account for gaze shifts)'''
        spatialOrigin = spatialFieldSize + spatialHalfSize + 1
        
        input_x = np.zeros((tMax, 1))
        input_g = np.zeros((tMax, 1))
        
        #need to check time indexing (from 0 or 1?)
        input_g[range(0,tMax)] = stimStrength_g_base
        input_x[range(0,tFixEnd)] = stimStrength_x
        if c == VALID_CUE:
            stimPos_v = [0, targetEccentricity, targetEccentricity]
            stimSize_v = [fixationSize, cueSize, targetSize] 
            stimTimes_v = np.array([[1, tFixEnd], [tCueStart, tCueEnd], [tTargetStart, tTargetEnd]])
            input_g[range(tToneBoostStart,tToneBoostEnd)] = stimStrength_g_base + stimStrength_g_tone
        elif c == INVALID_CUE:
            stimPos_v = [0, -targetEccentricity, targetEccentricity]
            stimSize_v = [fixationSize, cueSize, targetSize]
            stimTimes_v = np.array([[1, tFixEnd], [tCueStart, tCueEnd], [tTargetStart, tTargetEnd]])
            input_g[range(tToneBoostStart,tToneBoostEnd)] = stimStrength_g_base + stimStrength_g_tone
        elif c == DOUBLE_CUE:
            stimPos_v = [0, -targetEccentricity, targetEccentricity, targetEccentricity]
            stimSize_v = [fixationSize, cueSize, cueSize, targetSize] 
            stimTimes_v = np.array([[1, tFixEnd], [tCueStart, tCueEnd], [tCueStart, tCueEnd], [tTargetStart, tTargetEnd]])
            input_g[range(tToneBoostStart,tToneBoostEnd)] = stimStrength_g_base + stimStrength_g_tone
        elif c == TONE_CUE:
            stimPos_v = [0, targetEccentricity]
            stimSize_v = [fixationSize, targetSize]
            stimTimes_v = np.array([[1, tFixEnd], [tTargetStart, tTargetEnd]])
            input_g[range(tToneBoostStart,tToneBoostEnd)] = stimStrength_g_base + stimStrength_g_tone
        elif c == NO_CUE:
            stimPos_v = [0, targetEccentricity]
            stimSize_v = [fixationSize, targetSize]
            stimTimes_v = np.array([[1, tFixEnd], [tTargetStart, tTargetEnd]])
        elif c == CVALID_CUE:
            stimPos_v = [0, targetEccentricity, targetEccentricity]
            stimSize_v = [fixationSize, cueSize, targetSize] 
            stimTimes_v = np.array([[1, tFixEnd], [tCueStart, tCueEnd], [tTargetStart, tTargetEnd]])
            input_g[range(tToneBoostStart,tToneBoostEnd)] = stimStrength_g_base + stimStrength_g_tone
        elif c == CINVALID_CUE:
            stimPos_v = [0, -targetEccentricity, targetEccentricity]
            stimSize_v = [fixationSize, cueSize, targetSize] 
            stimTimes_v = np.array([[1, tFixEnd], [tCueStart, tCueEnd], [tTargetStart, tTargetEnd]])
            input_g[range(tToneBoostStart,tToneBoostEnd)] = stimStrength_g_base + stimStrength_g_tone
        elif c == CDOUBLE_CUE:
            stimPos_v = [0, -targetEccentricity, targetEccentricity, targetEccentricity]
            stimSize_v = [fixationSize, cueSize, cueSize, targetSize] 
            stimTimes_v = np.array([[1, tFixEnd], [tCueStart, tCueEnd], [tCueStart, tCueEnd], [tTargetStart, tTargetEnd]])
            input_g[range(tToneBoostStart,tToneBoostEnd)] = stimStrength_g_base + stimStrength_g_tone
        elif c ==  CTONE_CUE:
            stimPos_v = [0, targetEccentricity]
            stimSize_v = [fixationSize, targetSize]
            stimTimes_v = np.array([[1, tFixEnd], [tTargetStart, tTargetEnd]])
            input_g[range(tToneBoostStart,tToneBoostEnd)] = stimStrength_g_base + stimStrength_g_tone
        elif c == CNO_CUE:
            stimPos_v = [0, targetEccentricity]
            stimSize_v = [fixationSize, targetSize]
            stimTimes_v = np.array([[1, tFixEnd], [tTargetStart, tTargetEnd]])                    
        
        '''% generate actual stimuli'''
        nStimuli_v = np.size(stimPos_v)
        
        stimuli_v = np.zeros((nStimuli_v, 3*spatialFieldSize))
        stimWeights_v = np.zeros((tMax, nStimuli_v))

        for k in range(nStimuli_v):
            if stimPos_v[k] >= 0:
                stimStart = stimPos_v[k] - np.ceil((stimSize_v[k]-1)/2) + spatialOrigin
                stimEnd = stimPos_v[k] + np.floor((stimSize_v[k]-1)/2) + spatialOrigin
            else:
                stimStart = stimPos_v[k] - np.floor((stimSize_v[k]-1)/2) + spatialOrigin
                stimEnd = stimPos_v[k] + np.ceil((stimSize_v[k]-1)/2) + spatialOrigin

            spatialInput = np.zeros((1, 3*spatialFieldSize))
            spatialInput[0,range(int(stimStart),int(stimEnd))] = 1
            stimuli_v[k, :] = spatialInput
            stimWeights_v[range(stimTimes_v[k, 0]-1,stimTimes_v[k, 1]-2), k] = 1
        
        saccStartTimesTemp = np.zeros((nRepeats))
        saccEndTimesTemp = np.zeros((nRepeats))
        saccMetricsTemp = np.zeros((nRepeats))


        '''%% loop over repetitions'''
        for k in range(nRepeats):
            
            gazeDirection = spatialOrigin
            saccadeInProgress = False
            saccMetricsIntegrator = 0
            gazeChangeCounter = 0
            
            shiftedStimuli_v = stimuli_v[:, range(int(gazeDirection-spatialHalfSize-1),int(gazeDirection+spatialHalfSize))]
            convStimuli_v = conv2(shiftedStimuli_v, kernel_v, 'same')
            
            '''%% update COSIVINA noise parameters -- scale by tau'''
            '''%multiplying by tau to move from in-line params to cosivina params'''
            sim.setElementParameters(['noise kernel a'], ['amplitude'], [q_a[a]*tau_a]) 
            sim.setElementParameters(['noise kernel s'], ['amplitude'], [q_s[a]*tau_s]) 
            sim.setElementParameters('noise r', 'amplitude', q_r*tau_s) 
            sim.setElementParameters('noise x', 'amplitude', q_x*tau_a) 
            sim.setElementParameters('noise g', 'amplitude', q_x*tau_a)               

            '''%% set up handles/structures to store history '''
            if storeHistory:
                history_a = np.zeros((tMax, spatialFieldSize))
                history_s = np.zeros((tMax, spatialFieldSize))
                history_r = np.zeros((tMax, 1))
                history_x = np.zeros((tMax, 1))
                history_g = np.zeros((tMax, 1))                  
            
            '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % actual trial starting here %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% initialize sim and field values'''
            
            # turn on interactive plotting
            plt.ion()

            sim.init()
            sim.t=0
            
            # prepare axes
            if (k==0) and (j==0) and (mode == 1):
                fig, axes = plt.subplots(3, 1)
            
            if mode == 1:
                for i in range(3):
                    axes[i].cla()
                    axes[i].set_xlim(0, spatialFieldSize)
                    axes[i].set_ylim(-15, 15)
                    axes[i].set_ylabel('activation')
                axes[2].set_xlabel('feature space')
 
                # plot initial state
                x = np.arange(spatialFieldSize)
                plot_stim, = axes[0].plot(x, sim.getComponent('input_v', 'output')[0], color='g')
                plot_a, = axes[1].plot(x, sim.getComponent('field a', 'activation')[0], color='b')
                plot_a_out, = axes[1].plot(x, sim.getComponent('field a', 'output')[0], color='r')
                plot_s, = axes[2].plot(x, sim.getComponent('field s', 'activation')[0], color='b')
                plot_s_out, = axes[2].plot(x, sim.getComponent('field s', 'output')[0], color='r')

            '''%% initialize fields'''
            handle_a = sim.getElement('field a')
            handle_a.activation[:] = h_a
            handle_s = sim.getElement('field s')
            handle_s.activation[:] = h_s
            handle_r = sim.getElement('r')
            handle_r.activation[:] = h_r
            handle_x = sim.getElement('x')
            handle_x.activation[:] = h_x
            handle_g = sim.getElement('g')
            handle_g.activation[:] = h_g
            handle_input_v = sim.getElement('input_v')
            handle_input_x = sim.getElement('input_x')
            handle_input_g = sim.getElement('input_g')

            while sim.t <= tMax:

                t = int(sim.t)
                
                '''% determine start of saccade and suppress input'''
                if not saccadeInProgress and handle_r.output >= theta_saccStart:
                    gazeChangeCounter = gazeChangeCounter + 1
                    saccStartTimesTemp[k] = t
                    saccadeInProgress = True
                
                '''% determine end of saccade and update input'''
                if saccadeInProgress and handle_r.output < theta_saccEnd:
                    saccEndTimesTemp[k] = t
                    saccMetricsTemp[k] = saccMetricsIntegrator
                    saccMetricsIntegrator = 0
                    if breakOnFirstSaccade:
                        break                       
                    gazeDirection = gazeDirection + np.round(saccMetricsTemp[k])
                    shiftedStimuli_v = stimuli_v[:, range(int(gazeDirection-spatialHalfSize-1),int(gazeDirection+spatialHalfSize))]
                    convStimuli_v = conv2(shiftedStimuli_v, kernel_v, 'same')
                    saccadeInProgress = False
                
                '''% accumulate eye movement command'''
                tmp = np.zeros(spatialFieldSizeT)
                tmp[0,:] = range(int(-spatialHalfSize-1),int(spatialHalfSize))
                tmp2 = handle_s.output * tmp
                saccMetricsIntegrator = saccMetricsIntegrator + c_sacc * np.sum(tmp2)
                
                '''% update inputs'''
                if saccadeInProgress:
                    input_vTmp = 0
                else:
                    # input_vTmp = np.sum(convStimuli_v * np.matlib.repmat(np.transpose(stimWeights_v[t, :]), spatialFieldSize, 1))
                    tmp3=np.transpose(np.matlib.repmat(stimWeights_v[t, :], spatialFieldSize, 1))
                    input_vTmp = np.sum(convStimuli_v * tmp3, 0)
                                                                
                handle_input_v.output[:] = input_vTmp
                handle_input_x.output[:] = input_x[t]
                handle_input_g.output[:] = input_g[t]
                
                '''%% store activation history'''
                if storeHistory:
                    history_a[t, :] = handle_a.activation
                    history_s[t, :] = handle_s.activation
                    history_r[t] = handle_r.activation
                    history_x[t] = handle_x.activation
                    history_g[t] = handle_g.activation
 
                '''%% COSIVINA step and manage GUI'''
                if mode == 1 and (t % gui_speed)==0:
                    plot_stim.set_ydata(handle_input_v.output)
                    plot_a.set_ydata(handle_a.activation)
                    plot_a_out.set_ydata(handle_a.output*10)
                    plot_s.set_ydata(handle_s.activation)
                    plot_s_out.set_ydata(handle_s.output*10)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                
                sim.step()
                               
            if printTrials:
                print('Run: '+str(k+1)+', Condition: '+str(c)+', RT: ' \
                    +str(saccEndTimesTemp[k] - tTargetStart + saccadeLatencyOffset)+', Metrics: ' \
                    +str(saccMetricsTemp[k]))

        
        saccStartTimes[a, j, :] = saccStartTimesTemp[:]
        saccEndTimes[a, j, :] = saccEndTimesTemp[:]
        saccMetrics[a, j, :] = saccMetricsTemp[:]
        
                

'''%% analysis of results %%
%%%%%%%%%%%%%%%%%%%%%%%%%'''

for a in range(nAges):
    for c in range(nConditions):
        ratesCorrect[c, a] = sum(saccMetrics[a, c, :] > 0)/nRepeats
        RTs = saccEndTimes[a, c, :] - tTargetStart + saccadeLatencyOffset
        validTrials = np.zeros((np.size(RTs)))
        for t in range(np.size(RTs)):
            if (saccMetrics[a, c, t] > 0) and (RTs[t] >= minSaccLatency) and (RTs[t] <= maxSaccLatency):
                validTrials[t] = 1
        latenciesCorrect[c, a] = np.mean(RTs[np.nonzero(RTs*validTrials)])
        
        
conditionLabels = ['double', 'invalid', 'none', 'tone', 'valid', 'doubleC', 'invalidC', 'noneC', 'toneC', 'validC']
   
if plotLatencies:   
    # %% latency data from experiments...  

    # %2015 IOWA paper
    if experi == 1:
        data = [[240.1056851, 309.2475139, 291.9686829, 285.440822, 224.8424472, math.nan, math.nan, math.nan, math.nan, math.nan], \
                [236.85021, 318.2754641, 269.1083238, 266.9803054, 187.9746586, math.nan, math.nan, math.nan, math.nan, math.nan], \
                [219.6434554, 266.5036194, 251.0644456, 244.7092378, 181.2987801, math.nan, math.nan, math.nan, math.nan, math.nan]]     
    
    # %2022 IOWA-C paper
    if experi == 2 or experi == 5: 
        data = [[265.9371, 321.1029, 304.1788, 290.24, 239.6734, 396.5122, 418.1626, 422.3075, 406.4922, 340.4153], \
                [241.1271, 302.5485, 283.9948, 274.29, 194.9691, 371.7753, 417.3871, 459.3784, 436.8378, 341.3544], \
                [256.9689, 307.043, 276.8283, 269.946, 217.4912, 361.7011, 415.1425, 491.8354, 432.7396, 385.617]]    
    
    # %longitudinal VWM cohort 1
    if experi == 3:
        data = [[166.675463, 137.3491182, 146.7720085, math.nan, 141.4093446, 252.8468271, 280.9752722, 269.9405462, math.nan, 192.256769], \
                [99.27460317, 122.1689508, 95.5796979, math.nan, 109.5492063, 210.3142083, 249.448254, 217.4628533, math.nan, 203.3783972], \
                [82.92933333, 101.2008052, 70.42854915, math.nan, 71.77986928, 171.2466131, 176.4784127, 152.8853815, math.nan, 143.8658046]]    

    # %longitudinal VWM cohort 2
    if experi == 4: 
        data = [[90.71501984, 111.7006647, 86.72520525, math.nan, 96.76909722, 172.8416667, 198.8265997, 164.3876812, math.nan, 136.8728828], \
                [59.14555556, 90.08320106, 97.06880342, math.nan, 68.76767677, 125.0074786, 160.2577228, 143.7532922, math.nan, 98.02174603], \
                [75.81421958, 95.61904762, 96.59393939, math.nan, 120.8240741, 152.3113445, 164.4778613, 140.2676367, math.nan, 89.52460317]]                 
        
    data = np.transpose(data)    
    dataIOWA = data[0:5,:]
    dataIOWAC = data[5:10,:]

    dIOWA = np.empty((N_CONDITIONS, N_AGES))
    dLabels = ["" for x in range(N_CONDITIONS)]
    for a in range(nAges):
        for c in range(nConditions):
            dIOWA[c,a] = data[conditions[c],a]
            dLabels[c] = conditionLabels[c]
                
    plt.figure()
    for a in range(nAges):
        ax = plt.subplot(1, nAges, a+1)
        ax.plot(latenciesCorrect[:,a], color='b')
        # ax.plot([1, 2], latenciesCorrect, color='b')
        ax.set_title('Latencies for Age Group'+str(ages[a]+1))
        ax.set_xlabel('Conditions')
        if a == 0:
            ax.set_ylabel('Latency')
        ax.set_xticks(range(nConditions))    
        ax.set_xticklabels(dLabels)
        # ax = plt.subplot(1, nAges, 2)
        ax.plot(dIOWA[:,a], color='r')
    plt.show()

if plotErrorRates:
    # %% accuracy data from experiments...
    
    # %2015 IOWA paper
    if experi == 1: 
        dataAcc = [[0.820843508, 0.597209503, 0.948544511, 0.930458893, 0.962408425, math.nan, math.nan, math.nan, math.nan, math.nan], \
                   [0.788919414, 0.456127206, 0.963245088, 0.963331807, 0.984661172, math.nan, math.nan, math.nan, math.nan, math.nan], \
                   [0.7495116, 0.437601981, 0.97275641, 0.981913919, 0.9628663, math.nan, math.nan, math.nan, math.nan, math.nan]]     
    
    # %2022 IOWA-C paper
    if experi == 2 or experi == 5: 
        dataAcc = [[0.8778, 0.6732, 0.9879, 0.9821, 0.9757, 0.8965, 0.834, 0.9817, 0.9786, 0.9761], \
                   [0.7895, 0.4903, 0.9854, 0.9848, 0.9839, 0.9239, 0.7874, 0.9938, 0.9589, 0.9926], \
                   [0.913, 0.6646, 0.9604, 0.9945, 0.9952, 0.9248, 0.8801, 0.9895, 0.9744, 0.9678]]
    
    # %longitudinal VWM cohort 1
    if experi == 3: 
        dataAcc = [[0.842540755, 0.832979408, 0.971803351, math.nan, 0.963888889, 0.866001716, 0.872844394, 0.913745591, math.nan, 0.927941176], \
                   [0.934620596, 0.891308556, 0.983168796, math.nan, 0.993902439, 0.949177313, 0.945009921, 0.966298877, math.nan, 0.992073171], \
                   [0.956682028, 0.814025043, 1, math.nan, 0.984677419, 0.981989247, 0.940347019, 0.965239375, math.nan, 0.994623656]] 

    # %longitudinal VWM cohort 2
    if experi == 4: 
        dataAcc = [[0.958296638, 0.825667004, 0.972635934, math.nan, 0.993380615, 0.926174889, 0.899349882, 0.940602837, math.nan, 0.97281746], \
                   [0.949794239, 0.849441505, 0.99218107, math.nan, 0.996153846, 0.961111111, 0.880555556, 0.973662551, math.nan, 0.959259259], \
                   [0.946052632, 0.877631579, 1, math.nan, 1, 0.974269006, 0.954824561, 0.984210526, math.nan, 1]]

    dataAcc = np.transpose(dataAcc)   
    dataAccIOWA = dataAcc[0:5,:]
    dataAccIOWAC = dataAcc[5:10,:]

    dAIOWA = np.empty((N_CONDITIONS, N_AGES))
    dLabels = ["" for x in range(N_CONDITIONS)]
    for a in range(nAges):
        for c in range(nConditions):
            dAIOWA[c,a] = dataAcc[conditions[c],a]
            dLabels[c] = conditionLabels[c]
               
    plt.figure()
    for a in range(nAges):
        ax = plt.subplot(1, nAges, a+1)
        ax.plot(ratesCorrect[:,a], color='b')
        # ax.plot([1, 2], latenciesCorrect, color='b')
        ax.set_title('PercentCorrect for Age Group'+str(ages[a]+1))
        ax.set_xlabel('Conditions')
        if a == 0:
            ax.set_ylabel('Percent Correct')
        ax.set_xticks(range(nConditions))    
        ax.set_xticklabels(dLabels)
        # ax = plt.subplot(1, nAges, 2)
        ax.plot(dAIOWA[:,a], color='r')
    plt.show()
   

if saveResults:
    varsToSave = ['ratesCorrect', 'latenciesCorrect', 'saccStartTimes', 'saccEndTimes', 'saccMetrics']
    d = {key: globals()[key] for key in varsToSave}
    savemat(resultBaseFilename, d)
   

elapsed = time.time() - starttime
print('Elapsed Time: '+str(elapsed))    

# no numba elapsed = 5722.8 sec; 4930 sec; 5248 sec; 5460 sec
# numba elapsed = 4178.4

