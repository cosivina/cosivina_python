from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

mexicanHatKernel2DSpec = [
    ('size', sizeTupleType),
    ('sigmaExcX', floatType),
    ('sigmaExcY', floatType),
    ('amplitudeExc', floatType),
    ('sigmaInhX', floatType),
    ('sigmaInhY', floatType),
    ('amplitudeInh', floatType),
    ('circularX', boolType),
    ('circularY', boolType),
    ('normalized', boolType),
    ('cutoffFactor', floatType),
    ('kernelRangeExcX', intArrayType),
    ('kernelRangeExcY', intArrayType),
    ('kernelRangeInhX', intArrayType),
    ('kernelRangeInhY', intArrayType),
    ('kernelExcX', arrayType1D),
    ('kernelExcY', arrayType1D),
    ('kernelInhX', arrayType1D),
    ('kernelInhY', arrayType1D),
    ('output', arrayType2D)
]

@jitclass(elementSpec + mexicanHatKernel2DSpec)
class MexicanHatKernel2D(Element):
    ''' Connective element performing 2D convolution with a
    difference-of-Gaussians kernel. '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1), sigmaExcY = 1., sigmaExcX = 1.,
            amplitudeExc = 0., sigmaInhY = 1., sigmaInhX = 1.,
            amplitudeInh = 0., circularY = True, circularX = True,
            normalized = True, cutoffFactor = 5.):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the input and output.
            sigmaExcY (float): Width parameter of the excitatory Gaussian
                component of the kernel in the vertical dimension.
            sigmaExcX (float): Same for horizontal dimension.
            amplitudeExc (float): Amplitude of the excitatory component.
            sigmaInhY (float): Width parameter of the inhibitory Gaussian
                component of the kernel in the vertical dimension.
            sigmaInhX (float): Same for horizontal dimension.
            amplitudeInh (float): Amplitude of the inhibitory component.
            circularY (bool): Flag indicating whether convolution is
                circular in vertical dimension.
            circularX (bool): Same for horizontal dimension.
            normalized (bool): Flag indicating whether Gaussian
                components are normalized before scaling with amplitude.
            cutoffFactor (float): Multiple of the greater sigma value
                at which the kernel is truncated for each dimension.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED,
                'sigmaExcY': PS_INIT_STEP_REQUIRED,
                'sigmaExcX': PS_INIT_STEP_REQUIRED,
                'amplitudeExc': PS_INIT_STEP_REQUIRED,
                'sigmaInhY': PS_INIT_STEP_REQUIRED,
                'sigmaInhX': PS_INIT_STEP_REQUIRED,
                'amplitudeInh': PS_INIT_STEP_REQUIRED,
                'circularY': PS_INIT_STEP_REQUIRED,
                'circularX': PS_INIT_STEP_REQUIRED,
                'normalized': PS_INIT_STEP_REQUIRED,
                'cutoffFactor': PS_INIT_STEP_REQUIRED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.size = size
        self.sigmaExcY = sigmaExcY
        self.sigmaExcX = sigmaExcX
        self.amplitudeExc = amplitudeExc
        self.sigmaInhY = sigmaInhY
        self.sigmaInhX = sigmaInhX
        self.amplitudeInh = amplitudeInh
        self.circularY = circularY
        self.circularX = circularX
        self.normalized = normalized
        self.cutoffFactor = cutoffFactor


    def init(self):
        self.kernelRangeExcY = computeKernelRange(
                (self.amplitudeExc != 0) * self.sigmaExcY, self.cutoffFactor,
                self.size[0], self.circularY);
        self.kernelRangeExcX = computeKernelRange(
                (self.amplitudeExc != 0) * self.sigmaExcX, self.cutoffFactor,
                self.size[1], self.circularX);
        self.kernelExcY = self.amplitudeExc * gauss(
                np.arange(-self.kernelRangeExcY[1], self.kernelRangeExcY[0] + 1),
                0, self.sigmaExcY, self.normalized)
        self.kernelExcX = gauss(
                np.arange(-self.kernelRangeExcX[1], self.kernelRangeExcX[0] + 1),
                0, self.sigmaExcX, self.normalized)
        self.kernelRangeInhY = computeKernelRange(
                (self.amplitudeInh != 0) * self.sigmaInhY, self.cutoffFactor,
                self.size[0], self.circularY);
        self.kernelRangeInhX = computeKernelRange(
                (self.amplitudeInh != 0) * self.sigmaInhX, self.cutoffFactor,
                self.size[1], self.circularX);
        self.kernelInhY = self.amplitudeInh * gauss(
                np.arange(-self.kernelRangeInhY[1], self.kernelRangeInhY[0] + 1),
                0, self.sigmaInhY, self.normalized)
        self.kernelInhX = gauss(
                np.arange(-self.kernelRangeInhX[1], self.kernelRangeInhX[0] + 1),
                0, self.sigmaInhX, self.normalized)
        self.output = np.zeros(self.size)


    def step(self, time, deltaT):
        inputT = self.inputs[0].transpose()

        if self.circularY:
            convExc = parCircConv(inputT, self.kernelExcY, self.kernelRangeExcY)
            convInh = parCircConv(inputT, self.kernelInhY, self.kernelRangeInhY)
        else:
            convExc = parLinearConv(inputT, self.kernelExcY, self.kernelRangeExcY)
            convInh = parLinearConv(inputT, self.kernelInhY, self.kernelRangeInhY)
        convExc = convExc.transpose()
        convInh = convInh.transpose()

        if self.circularX:
            self.output[:][:] = parCircConv(convExc, self.kernelExcX, self.kernelRangeExcX) \
                    - parCircConv(convInh, self.kernelInhX, self.kernelRangeInhX)
        else:
            self.output[:][:] = parLinearConv(convExc, self.kernelExcX, self.kernelRangeExcX) \
                    - parLinearConv(convInh, self.kernelInhX, self.kernelRangeInhX)

