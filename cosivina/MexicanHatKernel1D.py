from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

mexicanHatKernel1DSpec = [
    ('size', sizeTupleType),
    ('sigmaExc', floatType),
    ('amplitudeExc', floatType),
    ('sigmaInh', floatType),
    ('amplitudeInh', floatType),
    ('circular', boolType),
    ('normalized', boolType),
    ('cutoffFactor', floatType),
    ('kernelRange', intArrayType),
    ('kernel', arrayType1D),
    ('output', arrayType2D)
]

@jitclass(elementSpec + mexicanHatKernel1DSpec)
class MexicanHatKernel1D(Element):
    ''' Connective element performing 1D convolution with a
    difference-of-Gaussians kernel. '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1), sigmaExc = 1., amplitudeExc = 0.,
            sigmaInh = 1., amplitudeInh = 0., circular = True,
            normalized = True, cutoffFactor = 5.):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the input and output.
            sigmaExc (float): Width parameter of the excitatory Gaussian
                component of the kernel.
            amplitudeExc (float): Amplitude of the excitatory component.
            sigmaInh (float): Width parameter of the inhibitory Gaussian
                component of the kernel.
            amplitudeInh (float): Amplitude of the inhibitory component.
            circular (bool): Flag indicating whether convolution is
                circular.
            normalized (bool): Flag indicating whether Gaussian
                components are normalized before scaling with amplitude.
            cutoffFactor (float): Multiple of the greater sigma value
                at which the kernel is truncated.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED,
                'sigmaExc': PS_INIT_STEP_REQUIRED,
                'amplitudeExc': PS_INIT_STEP_REQUIRED,
                'sigmaInh': PS_INIT_STEP_REQUIRED,
                'amplitudeInh': PS_INIT_STEP_REQUIRED,
                'circular': PS_INIT_STEP_REQUIRED,
                'normalized': PS_INIT_STEP_REQUIRED,
                'cutoffFactor': PS_INIT_STEP_REQUIRED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.size = size
        self.sigmaExc = sigmaExc
        self.amplitudeExc = amplitudeExc
        self.sigmaInh = sigmaInh
        self.amplitudeInh = amplitudeInh
        self.circular = circular
        self.normalized = normalized
        self.cutoffFactor = cutoffFactor


    def init(self):
        self.kernelRange = computeKernelRange(
                max((self.amplitudeExc != 0) * self.sigmaExc,
                (self.amplitudeInh != 0) * self.sigmaInh),
                self.cutoffFactor, self.size[1], self.circular);
        # note flipped kernel ranges (because kernel is flipped again in convolution)
        self.kernel = self.amplitudeExc * \
                gauss(np.arange(-self.kernelRange[1], self.kernelRange[0] + 1),
                0, self.sigmaExc, self.normalized) - self.amplitudeInh * \
                gauss(np.arange(-self.kernelRange[1], self.kernelRange[0] + 1),
                0, self.sigmaInh, self.normalized)
        self.output = np.zeros(self.size)

    def step(self, time, deltaT):
        if self.circular:
            for i in range(self.size[0]):
                self.output[i][:] = circConv(self.inputs[0][i], self.kernel, self.kernelRange)
        else:
            for i in range(self.size[0]):
                self.output[i][:] = linearConv(self.inputs[0][i], self.kernel, self.kernelRange)

