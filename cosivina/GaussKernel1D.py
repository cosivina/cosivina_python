from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

gaussKernel1DSpec = [
    ('size', sizeTupleType),
    ('sigma', floatType),
    ('amplitude', floatType),
    ('circular', boolType),
    ('normalized', boolType),
    ('cutoffFactor', floatType),
    ('kernelRange', intArrayType),
    ('kernel', arrayType1D),
    ('output', arrayType2D)
]

@jitclass(elementSpec + gaussKernel1DSpec)
class GaussKernel1D(Element):
    ''' Connective element performing 1D convolution with a Gaussian
    kernel. '''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1), sigma = 1., amplitude = 0.,
            circular = True, normalized = True, cutoffFactor = 5.):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the input and output.
            sigma (float): Width parameter of the Gaussian kernel.
            amplitude (float): Amplitude of the kernel.
            circular (bool): Flag indicating whether convolution is
                circular.
            normalized (bool): Flag indicating whether kernel is
                normalized before scaling with amplitude.
            cutoffFactor (float): Multiple of sigma at which the kernel
                is truncated.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED,
                'sigma': PS_INIT_STEP_REQUIRED,
                'amplitude': PS_INIT_STEP_REQUIRED,
                'circular': PS_INIT_STEP_REQUIRED,
                'normalized': PS_INIT_STEP_REQUIRED,
                'cutoffFactor': PS_INIT_STEP_REQUIRED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'

        self.size = size
        self.sigma = sigma
        self.amplitude = amplitude
        self.circular = circular
        self.normalized = normalized
        self.cutoffFactor = cutoffFactor


    def init(self):
        self.kernelRange = computeKernelRange(
                (self.amplitude != 0) * self.sigma, self.cutoffFactor,
                self.size[1], self.circular);
        # note flipped kernel ranges (because kernel is flipped again in convolution)
        self.kernel = self.amplitude * gauss(np.arange(-self.kernelRange[1], self.kernelRange[0] + 1),
                0, self.sigma, self.normalized)
        self.output = np.zeros(self.size)


    def step(self, time, deltaT):
        if self.circular:
            for i in range(self.size[0]):
                self.output[i][:] = circConv(self.inputs[0][i], self.kernel, self.kernelRange)
        else:
            for i in range(self.size[0]):
                self.output[i][:] = linearConv(self.inputs[0][i], self.kernel, self.kernelRange)
        # self.output *= self.amplitude

