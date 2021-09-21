from cosivina.base import *

@njit
def wrap(x, bound = np.pi):
    return (x + bound) % (2 * bound) - bound


@njit
def sigmoid(x: np.ndarray, beta: float = 1., x0: float = 0.):
    return 1 / (1 + np.exp(-beta * (x-x0)))


@njit
def gauss(x: np.ndarray, mu: float = 0., sigma: float = 1.,
        normalized: bool = False):
    if sigma == 0:
        g = np.zeros(x.shape)
        g[x == mu] = 1.0
    else:
        g = np.exp( - (x - mu)**2 / (2 * sigma**2))
    if normalized and np.any(g):
        g /= np.sum(g)
    return g


@njit
def circularGauss(x: np.ndarray, mu: float = 0., sigma: float = 1.,
        normalized: bool = False):
    # x must be monotically increasing or decreasing, with a least two elements;
    # the distance between last and first element is assumed to be the same
    # as between the first two elements

    # absolute distance covered by x
    l = abs(np.take(x, -1) + np.take(x, 1) - 2 * np.take(x, 0))

    if sigma == 0:
        g = np.zeros(x.shape)
        g[wrap(x - mu, l/2) == 0] = 1.0
    else:
        g = np.exp(-0.5 * wrap(x - mu, l/2)**2 / sigma**2)
    if normalized and np.any(g):
        g /= np.sum(g)
    return g

@njit
def circularGauss2d(y: np.ndarray, x: np.ndarray, muY: float = 0., muX: float = 0.,
        sigmaY: float = 1., sigmaX: float = 1.,
        circularY: bool = True, circularX: bool = True, normalized: bool = False):
    # if mode.lower() != 'min' and mode.lower() != 'sum':
    #     raise ValueError('Mode must be either "min" or "sum".')
    if circularY:
        gaussY = circularGauss(y, muY, sigmaY, normalized)
    else:
        gaussY = gauss(y, muY, sigmaY, normalized)
    if circularX:
        gaussX = circularGauss(x, muX, sigmaX, normalized)
    else:
        gaussX = gauss(x, muX, sigmaX, normalized)
    return gaussY.reshape((gaussY.size, 1)) * gaussX.reshape((1, gaussX.size))


@njit
def computeKernelRange(sigma, cutoffFactor, fieldSize, circular = True):
    if circular:
        r = np.ceil(sigma * cutoffFactor)
        h = (fieldSize-1)/2
        return np.array([min(r, np.floor(h)), min(r, np.ceil(h))], dtype=intType)
    else:
        r = int(min(np.ceil(sigma * cutoffFactor), (fieldSize - 1)))
        return np.array([r, r], dtype=intType)


# explicit no-numba versions of functions to be used in elements
# implemented without numba
def makeParamDictNN(d):
    return d


def makeComponentListNN(l):
    return dict.fromkeys(l , 0)



# functions with numba-specific implementation, and no-numba alternatives
if options.useNumba:
    def getNumbaStatus():
        return True

    @njit
    def makeParamDict(d):
        td = typed.Dict.empty(stringType, intType)
        for key in d:
            td[str(key)] = intType(d[key])
        return td


    @njit
    def makeComponentList(l): # currently using dict because set of strings not yet supported
        td = typed.Dict.empty(stringType, intType)
        for e in l:
            td[str(e)] = intType(0)
        return td


    @njit
    def circConv(input, kernel, kernelRange):
        l = len(input)
        n = len(kernel)
        ext = np.concatenate((input[-kernelRange[0]:], input, input[:kernelRange[1]]))
        kflip = np.flip(kernel)
        c = np.zeros(l, dtype=floatType)
        for i in range(l):
            for j in range(n):
                c[i] += kflip[j] * ext[i+j]
        return c


    @njit
    def linearConv(input, kernel, kernelRange):
        l = len(input)
        n = len(kernel)
        kflip = np.flip(kernel)
        c = np.zeros(l, dtype=floatType)
        for i in range(l):
            d = kernelRange[0] - i
            for j in range(max(d, 0), min(d + l, n)):
                c[i] += kflip[j] * input[j - d]
        return c


    @njit(parallel = True)
    def parCircConv(input, kernel, kernelRange):
        s = input.shape
        n = len(kernel)

        kflip = np.flip(kernel)
        ext = np.concatenate((input[:, -kernelRange[0]:], input, input[:, :kernelRange[1]]), 1)

        c = np.zeros(s, dtype=floatType)
        for k in prange(s[0]):
            for i in prange(s[1]):
                for j in range(n):
                    c[k][i] += kflip[j] * ext[k][i+j]
        return c

    @njit(parallel = True)
    def parLinearConv(input, kernel, kernelRange):
        s = input.shape
        n = len(kernel)

        kflip = np.flip(kernel)
        c = np.zeros(s, dtype=floatType)
        for k in prange(s[0]):
            for i in range(s[1]):
                d = kernelRange[0] - i
                for j in range(max(d, 0), min(d + s[1], n)):
                    c[k][i] += kflip[j] * input[k][j - d]
        return c


    # # versions using numba implementation of np.convolve
    # # (these were faster on laptop with WinPython, but substantially
    # # slower on workstation with Anaconda, so not using them for now)
    # @njit
    # def circConv(input, kernel, kernelRange):
    #     ext = np.concatenate((input[-kernelRange[0]:], input, input[:kernelRange[1]]))
    #     s = kernelRange.sum()
    #     c = np.convolve(ext, kernel)[s : -s]
    #     return c
    #
    #
    # @njit
    # def linearConv(input, kernel, kernelRange):
    #     c = np.convolve(input, kernel)[kernelRange[1] : -kernelRange[0]]
    #     return c
    #
    #
    # @njit(parallel = True)
    # def parCircConv(input, kernel, kernelRange):
    #     sz = input.shape
    #     ext = np.concatenate((input[:, -kernelRange[0]:], input, input[:, :kernelRange[1]]), 1)
    #     s = kernelRange.sum()
    #
    #     c = np.zeros(sz, dtype=floatType)
    #     for k in prange(sz[0]):
    #         c[k] = np.convolve(ext[k], kernel)[s : -s]
    #     return c
    #
    #
    # @njit(parallel = True)
    # def parLinearConv(input, kernel, kernelRange):
    #     sz = input.shape
    #
    #     c = np.zeros(sz, dtype=floatType)
    #     for k in prange(sz[0]):
    #         c[k] = np.convolve(input[k], kernel)[kernelRange[0] : -kernelRange[1]]
    #     return c


else:
    def getNumbaStatus():
        return False

    def makeParamDict(d):
        return d


    def makeComponentList(l):
        return dict.fromkeys(l , 0)


    def circConv(input, kernel, kernelRange):
        return np.convolve(np.pad(input, kernelRange, mode = 'wrap'), kernel, 'valid')


    def linearConv(input, kernel, kernelRange):
        return np.convolve(np.pad(input, kernelRange), kernel, 'valid')


    def parCircConv(input, kernel, kernelRange):
        s = input.shape
        c = np.zeros(s, dtype=floatType)
        for i in range(s[0]):
            c[i][:] = np.convolve(np.pad(input[i], kernelRange, mode = 'wrap'), kernel, 'valid')
        return c


    def parLinearConv(input, kernel, kernelRange):
        s = input.shape
        c = np.zeros(s, dtype=floatType)
        for i in range(s[0]):
            c[i][:] = np.convolve(np.pad(input[i], kernelRange), kernel, 'valid')
        return c

