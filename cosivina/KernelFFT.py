from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec


# element implemented without numba, since numba does not currently
# offer fft functions
class KernelFFT(Element):
    ''' Connective element performing convolution with a
    difference-of-Gaussians kernel with global component, using
    transformation into Fourier space.

    This element does not have a numba implementation, so the no-numba
    version is always used.'''
    initElement = Element.__init__

    def __init__(self, label, size = (1, 1), sigmaExc = None, amplitudeExc = 0.,
            sigmaInh = None, amplitudeInh = 0., amplitudeGlobal = 0.,
            circular = None, normalized = True, paddingFactor = 5.):
        '''
        Args:
            label (str): Element label.
            size (tuple of int): Size of the input and output.
            sigmaExc (list of floats): Width parameters of the
                excitatory Gaussian components of the kernel in each
                dimension.
            amplitudeExc (float): Amplitude of the excitatory component.
            sigmaInh (list of floats): Width parameters of the
                inhibitory Gaussian components of the kernel in each
                dimension.
            amplitudeInh (float): Amplitude of the inhibitory component.
            amplitudeGlobal (float): Amplitude of the global component.
            circularY (list of bools): Flags indicating whether
                convolution is circular in each dimension. Currently
                only circular convolutions are implemented.
            normalized (bool): Flag indicating whether Gaussian
                components are normalized before scaling with amplitude.
            paddingFactor (float): Currently not used.
        '''
        self.initElement(label)
        self.parameters = makeParamDictNN({
                'size': PS_FIXED,
                'sigmaExc': PS_INIT_STEP_REQUIRED,
                'amplitudeExc': PS_INIT_STEP_REQUIRED,
                'sigmaInh': PS_INIT_STEP_REQUIRED,
                'amplitudeInh': PS_INIT_STEP_REQUIRED,
                'amplitudeGlobal': PS_INIT_STEP_REQUIRED,
                'circular': PS_INIT_STEP_REQUIRED,
                'normalized': PS_INIT_STEP_REQUIRED,
                'paddingFactor': PS_INIT_STEP_REQUIRED
            })
        self.components = makeComponentListNN(['output', 'kernelFFT'])
        self.defaultOutputComponent = 'output'

        self.size = size
        nDim = min(np.sum(np.array(self.size) > 1), 1)

        if not sigmaExc:
            sigmaExc = np.ones(nDim)
        elif np.isscalar(sigmaExc):
            sigmaExc = [sigmaExc]
        if not sigmaInh:
            sigmaInh = np.ones(nDim)
        elif np.isscalar(sigmaInh):
            sigmaInh = [sigmaInh]
        if not circular:
            circular = np.ones(nDim, dtype=bool)
        elif np.isscalar(circular):
            circular = [circular]

        # using column vector (2D array) for compatibility with Matlab
        self.sigmaExc = np.array(sigmaExc).reshape((-1, 1))
        self.amplitudeExc = amplitudeExc
        self.sigmaInh = np.array(sigmaInh).reshape((-1, 1))
        self.amplitudeInh = amplitudeInh
        self.amplitudeGlobal = amplitudeGlobal
        self.circular = np.array(circular, dtype=bool).reshape((-1, 1))
        self.normalized = normalized
        self.paddingFactor = paddingFactor

        # check consistency of arguments
        if not self.circular.all():
            raise ValueError('Currently only circular convolutions are supported.')


    def init(self):
        nDim = max(np.sum(np.array(self.size) > 1), 1)
        if nDim == 2:
            sz = self.size[0]
        else:
            sz = np.max(np.array(self.size))

        rng = np.concatenate((np.arange(0, np.floor(sz/2) + 1),
                np.arange(-np.floor((sz-1)/2), 0)))
        kernelExc = self.amplitudeExc * gauss(rng, 0, self.sigmaExc[0, 0], self.normalized)
        kernelInh = self.amplitudeInh * gauss(rng, 0, self.sigmaInh[0, 0], self.normalized)

        if nDim == 2:
            sz = self.size[1]
            rng = np.concatenate((np.arange(0, np.floor(sz/2) + 1),
                np.arange(-np.floor((sz-1)/2), 0)))
            kernelExc = np.reshape(kernelExc, (self.size[0], 1)) \
                    * gauss(rng, 0, self.sigmaExc[1, 0], self.normalized)
            kernelInh = np.reshape(kernelInh, (self.size[0], 1)) \
                    * gauss(rng, 0, self.sigmaInh[1, 0], self.normalized)

        kernel = np.reshape(kernelExc - kernelInh + self.amplitudeGlobal, self.size)
        self.kernelFFT = np.fft.rfft2(kernel)

        self.output = np.zeros(self.size)


    def step(self, time, deltaT):
        self.output[:][:] = np.fft.irfft2(np.fft.rfft2(self.inputs[0]) * self.kernelFFT)
