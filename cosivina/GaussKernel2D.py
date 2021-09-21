from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

gaussKernel2DSpec = [
    ('size', sizeTupleType),
    ('sigmaX', floatType),
    ('sigmaY', floatType),
    ('amplitude', floatType),
    ('circularX', boolType),
    ('circularY', boolType),
    ('normalized', boolType),
    ('cutoffFactor', floatType),
    ('kernelRangeX', intArrayType),
    ('kernelRangeY', intArrayType),
    ('kernelX', arrayType1D),
    ('kernelY', arrayType1D),
    ('output', arrayType2D)
]

@jitclass(elementSpec + gaussKernel2DSpec)
class GaussKernel2D(Element):
    ''' Connective element performing 2D convolution with a Gaussian
    kernel. '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1), sigmaY = 1., sigmaX = 1.,
            amplitude = 0., circularY = True, circularX = True,
            normalized = True, cutoffFactor = 5.):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the input and output.
            sigmaY (float): Width parameter of the Gaussian kernel in
                the vertical dimension.
            sigmaX (float): Same for horizontal dimension.
            amplitude (float): Amplitude of the kernel.
            circularY (bool): Flag indicating whether convolution is
                circular in vertical dimension.
            circularX (bool): Same for horizontal dimension.
            normalized (bool): Flag indicating whether kernel is
                normalized before scaling with amplitude.
            cutoffFactor (float): Multiple of sigma at which the kernel
                is truncated for each dimension.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED,
                'sigmaY': PS_INIT_STEP_REQUIRED,
                'sigmaX': PS_INIT_STEP_REQUIRED,
                'amplitude': PS_INIT_STEP_REQUIRED,
                'circularY': PS_INIT_STEP_REQUIRED,
                'circularX': PS_INIT_STEP_REQUIRED,
                'normalized': PS_INIT_STEP_REQUIRED,
                'cutoffFactor': PS_INIT_STEP_REQUIRED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.size = size
        self.sigmaY = sigmaY
        self.sigmaX = sigmaX
        self.amplitude = amplitude
        self.circularY = circularY
        self.circularX = circularX
        self.normalized = normalized
        self.cutoffFactor = cutoffFactor


    def init(self):
        self.kernelRangeY = computeKernelRange(
                (self.amplitude != 0) * self.sigmaY, self.cutoffFactor,
                self.size[0], self.circularY);
        self.kernelRangeX = computeKernelRange(
                (self.amplitude != 0) * self.sigmaX, self.cutoffFactor,
                self.size[1], self.circularX);
        self.kernelY = self.amplitude * gauss(
                np.arange(-self.kernelRangeY[1], self.kernelRangeY[0] + 1),
                0, self.sigmaY, self.normalized)
        self.kernelX = gauss(
                np.arange(-self.kernelRangeX[1], self.kernelRangeX[0] + 1),
                0, self.sigmaX, self.normalized)
        self.output = np.zeros(self.size)


    def step(self, time, deltaT):
        inputT = self.inputs[0].transpose()

        if self.circularY:
            conv = parCircConv(inputT, self.kernelY, self.kernelRangeY)
        else:
            conv = parLinearConv(inputT, self.kernelY, self.kernelRangeY)
        conv = conv.transpose()

        if self.circularX:
            self.output[:][:] = parCircConv(conv, self.kernelX, self.kernelRangeX)
        else:
            self.output[:][:] = parLinearConv(conv, self.kernelX, self.kernelRangeX)

