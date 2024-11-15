import numpy as np
from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

coordinateTransformationSpec = [
    ('size', sizeTupleType),
    ('circular', boolType),
    ('normalized', boolType),
    ('cutoffFactor', floatType),
    ('kernelRange', intArrayType),
    ('kernel', arrayType1D),
    ('output', arrayType2D)
]

@jitclass(elementSpec + coordinateTransformationSpec)
class CoordinateTransformation(Element):
    ''' A element that performs the coordinate transformation that is neurally postulated to be realized
    through joint representation and diagonal sums. Currently implemented using convolution. '''
    initElement = Element.__init__

    def __init__(self, label, size=(1, 1), circular=False, normalized=True, cutoffFactor = 5., mode='same'):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the output.
            circular (bool): Flag indicating whether convolution is
                circular.
            normalized (bool): Flag indicating whether Gaussian
                components are normalized before scaling with amplitude.
            cutoffFactor (float): Multiple of the greater sigma value
                at which the kernel is truncated.
            mode (str): Mode of convolution, currently only the 'same' mode is defined.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
            'size': PS_FIXED,
            'circular': PS_INIT_STEP_REQUIRED,
            'normalized': PS_INIT_STEP_REQUIRED,
            'cutoffFactor': PS_INIT_STEP_REQUIRED
        })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.size = size
        self.circular = circular
        self.normalized = normalized
        self.cutoffFactor = cutoffFactor


    def init(self):
        # TODO: output dimensionality should depend on the mode
        self.output = np.zeros(self.size)

    def step(self, time, deltaT):
        if self.size[0] == 1:  # 1D
            if self.circular:
                raise ValueError('Currently only linear correlations are supported.')
            else:
                self.output[0] = np.convolve(self.inputs[0][0], self.inputs[1][0], mode='same')
        else:
            raise ValueError("unsupported dimensionality")
