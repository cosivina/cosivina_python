from cosivina.base import *
from cosivina.auxiliary import *
from cosivina.Element import Element, elementSpec

customStimulusSpec = [
    ('size', sizeTupleType),
    ('stimulusPattern', arrayType2D),
    ('output', arrayType2D)
]

@jitclass(elementSpec + customStimulusSpec)
class CustomStimulus(Element):
    ''' Fixed custom stimulus defined as a numpy array. '''
    initElement = Element.__init__

    def __init__(self, label, stimulusPattern = None):
        '''
        Args:
            label (str): Element label.
            stimulusPattern (1D or 2D numpy array): Full stimulus
                pattern.
        '''
        self.initElement(label)
        self.parameters = makeParamDict({
                'size': PS_FIXED
            })
        self.components = makeComponentList(['output'])
        self.defaultOutputComponent = 'output'
        if stimulusPattern is None:
            self.stimulusPattern = np.zeros((0, 0))
            self.size = (0, 0)
        else:
            nd = stimulusPattern.ndim
            s = stimulusPattern.shape
            if nd == 1:
                self.size = (1, s[0])
            else:
                self.size = (s[0], s[1])
            self.stimulusPattern = np.zeros(self.size)
            self.stimulusPattern[:][:] = stimulusPattern

    def init(self):
        self.output = self.stimulusPattern

    def step(self, time, deltaT):
        pass
