from cosivina.base import *
from cosivina.elements import *
import json
from scipy.io import savemat


class Simulator:
    '''
    Core class to create a neurodynamic architecture and simulate evolution
        of activation distributions over time.

    Attributes:
        tZero (float): Intial simulation time.
        deltaT (float): Duration of each simulation step.
        t (float): Current simulation time.
        initialized (bool): Initialization status.
        elementLabels (list of str): List of all element labels.

    Methods for creating architectures:
        addElement: Add a new element with connections.
        addConnection: Add connections between existing elements.
        addField: Add a new NeuralField, its History, and interaction kernel elements.

    Methods for running simulations:
        init: Initialize the simulation.
        step: Perform single simulation step.
        run: Run simulation for a specified time.
        close: Close all elements.

    Methods for accessing elements:
        isElements: Check if element label exists.
        getElement: Get reference to element.
        getComponent: Get reference to element component.
        setElementParameters: Change parameters of element(s).

    Methods for saving and loading:
        saveSettings: Save architecture to json file.
        loadSettings: Load parameters from json file.
    '''

    def __init__(self, tZero = 0., deltaT = 1., *, struct = None, file = ''):
        '''
        Constructor calls:
            Simulator(tZero, deltaT)
            Simulator(file='settings.json')
            Simulator(struct=jsonStruct)

        Args:
            tZero (float, optional): Time at initialization (in
                arbitrary units). Defaults to 0.
            deltaT (float, optional): Duration of each time steps (in
                arbitrary units). Defaults to 1.
            file (str): Filename for a settings file in json format.
            sruct (obj): Json struct to load settings from.
        '''
        self.elements = []
        self.elementDict = {}
        self.elementLabels = [] # redundant, included to match Matlab implementation
        self.targetDict = {} # list of elements receiving input from each element

        if not struct is None:
            self.fromStruct(struct)
        elif file:
            try:
                with open(file, 'r') as f:
                    j = json.load(f)
            except IOError:
                warning('Warning: Could not read json struct from file. ' \
                        'Simulator object  will be empty.')
            self.fromStruct(j['simulator'])
        else:
            self.tZero = tZero
            self.deltaT = deltaT

        self.initialized = False
        self.t = tZero

    def init(self):
        ''' Set time to tZero and (re-)initialize all elements.'''

        self.t = self.tZero
        for el in self.elements:
            el.init()

        self.refreshConnections()
        self.initialized = True

    def refreshConnections(self):
        ''' Reset references to input components in all elements. '''
        for el in self.elements:
            el.inputs.clear()
            for i in range(el.nInputs):
                ie = self.elementDict[el.inputElementLabels[i]]
                el.inputs.append(getattr(ie, el.inputComponents[i]))

    def step(self):
        ''' Perform one simulation step. '''
        self.t += self.deltaT
        for el in self.elements:
            el.step(self.t, self.deltaT)

    def close(self):
        ''' Close all elements.

        This is only required if there are elements that need to be
        closed after a simulation, e.g. because they access a file or
        connect to hardware. Otherwise, the method will have no effect.
        '''
        for el in self.elements:
            el.close()

    def run(self, tMax, initialize = False, closeWhenFinished = False):
        '''
        Run simulation up to a given simulation time.

        Args:
            tMax (float): Simulation time up to which simulation is
                run.
            initialize (bool, optional): (Re-)initialize the simulation
                before running. Defaults to False.
            closeWhenFinished (bool, optional): Close all elements after
                reaching tMax. Defaults to False.
        '''
        if not self.initialized or initialize:
            self.init()
        while self.t < tMax:
            self.step()
        if closeWhenFinished:
            self.close()

    def addElement(self, element, inputLabels = None, inputComponents = None,
            targetLabels = None, componentsForTargets = None):
        '''
        Add new element and connect it to existing architecture.

        Args:
            element (obj): The element to be added (typically created
                by calling its constructor).
            inputLabels (list of str, optional): Labels of existing
                elements from which the new element receives input.
                Defaults to None.
            inputLabels (list of str, optional): Components of input
                elements that are fed as inputs to the new element.
                Must have same length as inputLabels. If left empty,
                the defaultOutputComponent of each input element is
                used. Defaults to None.
            targetLabels (list of str, optional): Labels of existing
                elements that receive input from the new element.
                Defaults to None.
            inputLabels (list of str, optional): Components of the new
                element that will be fed as inputs to the target
                elements. Must have same length as targetLabels. If left
                empty, the defaultOutputComponent of the new element
                is used. Defaults to None.
        '''

        # if not isinstance(element, Element): # not currently supported for jitclass
        #     raise TypeError('Argument "element" must be an instance of a '\
        #             'subclass of Element.')
        label = element.label
        if not label:
            raise ValueError('Label of added element cannot be empty.')
        if label in self.elementDict:
            raise ValueError('Label "' + label \
                    + '" is already used in simulator object.')

        # make sure all arguments are lists of strings
        if inputLabels is None:
            inputLabels = []
        elif isinstance(inputLabels, str):
            inputLabels = [inputLabels]
        if isinstance(inputComponents, str):
            inputComponents = [inputComponents]
        elif not inputComponents:
            inputComponents = [''] * len(inputLabels)
        if targetLabels is None:
            targetLabels = []
        elif isinstance(targetLabels, str):
            targetLabels = [targetLabels]
        if isinstance(componentsForTargets, str):
            componentsForTargets = [componentsForTargets]
        elif not componentsForTargets:
            componentsForTargets = [''] * len(targetLabels)

        # check whether input/target lists have the same length
        if len(inputComponents) != len(inputLabels):
            raise ValueError('Argument "inputComponents" must have the same '\
                    'length as argument "inputLabels", or be empty.')
        if len(targetLabels) != len(componentsForTargets):
            raise ValueError('Argument "componentsForTargets" must have the '\
                    'same length as argument "targetLabels", or be empty.')

        # collect and check inputs
        inputElements = [];
        for i in range(len(inputLabels)):
            l = inputLabels[i]
            if not l in self.elementDict:
                raise ValueError('Element "' + l + '" requested as '\
                        'input for new element not found in simulator object.')
            c = inputComponents[i]
            el = self.elementDict[l]
            if not c:
                inputComponents[i] = el.defaultOutputComponent
            elif not c in el.components:
                raise ValueError('Invalid input component "' + c \
                        + '" requsted for input element "' + l + '".')

        # check outputs
        for i in range(len(targetLabels)):
            l = targetLabels[i]
            c = componentsForTargets[i]
            if not l in self.elementDict:
                raise ValueError('Element label "' + l + '" requested as '\
                        'target for new element not found in simulator object.')
            if not c:
                componentsForTargets[i] = element.defaultOutputComponent
            elif not c in element.components:
                raise ValueError('Invalid component "' + c + '" of new '\
                        'element requsted for target element "' + l + '".')

        # add element with inputs
        self.elements.append(element)
        self.elementLabels.append(label)
        self.elementDict[label] = element
        for i in range(len(inputLabels)):
            element.addInput(inputLabels[i], inputComponents[i])

        # add element as input for other elements
        for i in range(len(targetLabels)):
            el = self.elementDict[targetLabels[i]]
            el.addInput(label, componentsForTargets[i])

        # update targetDict
        self.targetDict[label] = targetLabels
        for il in inputLabels:
            self.targetDict[il].append(label)

        # re-connect if simulator is already initialized
        if self.initialized:
            self.refreshConnections()

    def addConnection(self, inputLabels, inputComponents, targetLabel):
        '''
        Add new connection(s) between existing elements.

        Args:
            inputLabels (list of str): Labels of existing
                elements from which new connections originate.
            inputComponents (list of str): Components of input
                elements that are fed as inputs to the target element. If
                not specified, the defaultOutputComponent of each input
                element is used. Must have same length as inputLabels.
            targetLabel (list of str): Label of existing element that
                is the target of all new connections.
        '''

        if isinstance(inputLabels, str):
            inputLabels = [inputLabels]
        if isinstance(inputComponents, str):
            inputComponents = [inputComponents]
        elif not inputComponents:
            inputComponents = [''] * len(inputLabels)
        if len(inputComponents) != len(inputLabels):
            raise ValueError('Argument "inputComponents" must have the same '\
                    'length as argument "inputLabels", or be empty.')

        # check target label
        if not isinstance(targetLabel, str):
            raise ValueError('Argument targetLabel must be a single string.')
        elif not targetLabel in self.elementDict:
            raise ValueError('Element label "' + targetLabel + '" requested '\
                    'as target for connection not found in simulator object.')

        # check input labels and components
        for i in range(len(inputLabels)):
            l = inputLabels[i]
            if not l in self.elementDict:
                raise ValueError('Element label "' + l + '" requested '\
                    'as input for connection not found in simulator object.')
            el = self.elementDict[l]
            c = inputComponents[i]
            if not c:
                inputComponents[i] = el.defaultOutputComponent
            elif not el.isComponent(c):
                raise ValueError('Invalid input component "' + c
                        + '" requsted for input element "' + l + '".')

        # create new connections
        target = self.elementDict[targetLabel]
        for i in range(len(inputLabels)):
            target.addInput(inputLabels[i], inputComponents[i])
            self.targetDict[inputLabels[i]].append(targetLabel)

        if self.initialized:
            self.refreshConnections()

    def addField(self, time_run, label, size=(1, 1), tau=10, h=-5, beta=2, amplitudeExc=0, amplitudeInh=0,
                 amplitudeGlobal=0, sigmaExc=4, sigmaInh=8, circular=True, normalized=True, cutoffFactor=3,
                 inputLabels=None, inputComponents=None, targetLabels=None, historyComponents="activation"):
        """
        wrapper to add the NeuralField, History, and the Kernel elements in one line.
        Currently the storingTimes of History is fixed to be every time step.

        Args:
            time_run (int): Simulation run time in msec, needed to initialize the History element with the right size.
            historyComponents (str): Specifies which components of the NeuralField should be recorded.
        """
        # NeuralField
        self.addElement(NeuralField("Field" + label, size=size, tau=tau, h=h, beta=beta),
                        inputLabels=inputLabels,
                        inputComponents=inputComponents,
                        targetLabels=targetLabels
                        )

        # History
        # TODO: set the storingTimes dynamically
        self.addElement(History("Hist" + label, size=size, storingTimes=[*range(int(time_run / self.deltaT))]),
                        inputLabels="Field" + label,
                        inputComponents=historyComponents
                        )

        # Interaction kernel
        if size == [1, 1]:  # 0D node
            self.addElement(ScaleInput("Kernel" + label, amplitude=amplitudeExc),
                            inputLabels="Field" + label,
                            targetLabels="Field" + label
                            )

        elif size[0] == 1:  # 1D
            self.addElement(LateralInteractions1D("Kernel" + label, size=size, sigmaExc=sigmaExc,
                                                  amplitudeExc=amplitudeExc, sigmaInh=sigmaInh,
                                                  amplitudeInh=amplitudeInh, amplitudeGlobal=amplitudeGlobal,
                                                  circular=circular, normalized=normalized, cutoffFactor=cutoffFactor),
                            inputLabels="Field" + label,
                            targetLabels="Field" + label
                            )

        else:  # 2D, use FFT convolution
            # TODO: possibility to specify different sigma for different dimensions
            self.addElement(KernelFFT("Kernel" + label, size=size, sigmaExc=[[sigmaExc], [sigmaExc]],
                                      amplitudeExc=amplitudeExc, sigmaInh=[[sigmaInh], [sigmaInh]],
                                      amplitudeInh=amplitudeInh, amplitudeGlobal=amplitudeGlobal,
                                      circular=[[circular], [circular]], normalized=normalized),
                            inputLabels="Field" + label,
                            targetLabels="Field" + label
                            )

    def isElement(self, elementLabel):
        ''' Check if element exists in architecture.

        Args:
            elementLabel (str): Potential element label.
        '''
        return elementLabel in self.elementDict

    def getElement(self, elementLabel):
        ''' Get reference to element in architecture.

        Returns None if no elements with the specified label exists.

        Args:
            elementLabel (str): Label of an existing element.
        '''
        if elementLabel in self.elementDict:
            return self.elementDict[elementLabel]

    def getComponent(self, elementLabel, componentName):
        ''' Get reference to an element's component.

        Args:
            elementLabel (str): Label of an existing element.
            componentName (str): A component of that element.
        '''
        if not elementLabel in self.elementDict:
            raise ValueError('No element "' + elementLabel
                    + '" in simulator object.')
        el = self.elementDict[elementLabel]
        if not el.isComponent(componentName):
            raise ValueError('Invalid component "' + componentName
                    + '" requested for element "' + elementLabel + '".')
        return getattr(el, componentName)

    def getElementParameter(self, elementLabel, parameterName):
        ''' Get a parameter value of an element.

        May return a copy or a reference, depending on the type of the
        parameter.

        Args:
            elementLabel (str): Label of an existing element.
            parameterName (str): A parameter of that element.
        '''
        if not elementLabel in self.elementDict:
            raise ValueError('No element "' + elementLabel
                    + '" in simulator object.')
        el = self.elementDict[elementLabel]
        if el.isParameter(parameterName):
            return getattr(el, parameterName)
        else:
            raise ValueError('Invalid parameter "' + parameterName
                    + '" requested for element "' + elementLabel + '".')

    def setElementParameters(self, elementLabels, parameterNames, newValues):
        ''' Set one or more elements' parameters to new values.

        Args:
            elementLabels (list of str): Labels of existing elements.
            parameterNames (list of str): Name of a parameter of each
                element. Must have same length as elementLabels.
            newValues (list of float/int/other): New values for each
                parameter. Must have same length as elementLabels.
        '''
        if isinstance(elementLabels, str):
            elementLabels = [elementLabels]
        if isinstance(parameterNames, str):
            parameterNames = [parameterNames]
        if not isinstance(newValues, list):
            newValues = [newValues]
        labelSet = set(elementLabels)

        n = len(parameterNames)
        if len(newValues) != n:
            raise ValueError('Arguments parameterNames and newValues '\
                    'must have the same number of elements.')

        if n > 1 and len(elementLabels) == 1:
            elementLabels *= n
        elif len(elementLabels) != n:
            raise ValueError('Arguments elementLabels and parameterNames '\
                    'must have the same length, or elementLabels must be '\
                    'a single string.')

        # check elements and parameters
        statusDict = dict.fromkeys(elementLabels, PS_CHANGEABLE)
        for i in range(n):
            l = elementLabels[i]
            p = parameterNames[i]
            if not l in self.elementDict:
                raise ValueError(f'Requested element "{l}" not found in '\
                        'simulator object.')
            el = self.elementDict[l]
            if not el.isParameter(p):
                raise ValueError (f'Invalid parameter "{p}" requested for '\
                        'element "{l}".')
            s = el.getParameterStatus(p)
            if s == PS_FIXED:
                raise ValueError(f'Parameter "{p}"" of element "{l}" cannot '\
                        'be changed.')
            else:
                statusDict[l] = max(statusDict[l], s)
            if not isinstance(newValues[i], (int, float)):
                raise TypeError('New parameter values must be numeric values.')

        # update parameters
        for i in range(n):
            el = self.elementDict[elementLabels[i]]
            setattr(el, parameterNames[i], newValues[i])

        # re-initialize elements and call step function as required
        if self.initialized:
            for l in statusDict:
                el = self.elementDict[l]
                s = statusDict[l]

                if s == PS_INIT_REQUIRED or s == PS_INIT_STEP_REQUIRED:
                    el.init()

                    # update inputs for all target elements of el
                    for tl in self.targetDict[l]:
                        te = self.elementDict[tl]
                        for i in range(te.nInputs):
                            if te.inputElementLabels[i] == l:
                                te.inputs[i] = getattr(el, te.inputComponents[i])

                if s == PS_INIT_STEP_REQUIRED:
                    el.step(self.t, self.deltaT)

    def saveSettings(self, filename):
        ''' Save architecture and parameters to json file.

        Args:
            filename (str): Valid file name.

        Returns:
            True for success, False otherwise..
        '''
        try:
            with open(filename, 'w+') as f:
                json.dump(self.toStruct(), f)
        except IOError as e:
            print('Could not write json struct to file: ' + e)
            return False
        return True

    def loadSettings(self, filename, parameters = 'changeable'):
        ''' Load parameters from json file, preserving architecture.

        Args:
            filename (str): Valid file name.
            parameters (str): Either 'changeable' or 'all'; the former
                will only overwrite parameters that do not have status
                fixed. Defaults to 'changeable'.

        Returns:
            True for success, False otherwise..
        '''

        changeableOnly = False;
        if parameters.lower() == 'changeable':
            changeableOnly = True
        elif parameters.lower() != 'all':
            raise ValueError('Argument "parameters" must be either of two '\
                    ' strings, "changeable" or "all".')

        try:
            with open(filename, 'r') as f:
                j = json.load(f)
        except IOError as e:
            print('Could not read json struct from file: ' + e)
            return False
        self.parametersFromStruct(j['simulator'], changeableOnly)
        return True

    def fromStruct(self, simStruct):
        self.elements.clear()
        self.elementDict.clear()
        self.targetDict.clear()

        self.deltaT = simStruct['deltaT']
        self.tZero = simStruct['tZero']
        nElements = simStruct['nElements']

        for i in range(nElements):
            elStruct = simStruct['elements'][i]

            # create element
            classStr = elStruct['class']
            constructor = globals()[classStr] # creates callable function
            elLabel = elStruct['label']
            el = constructor(label = elLabel)

            # set parameters
            elParam = elStruct['param']
            for name, value in elParam.items():
                p = getattr(el, name)
                if isinstance(p, np.ndarray):
                    va = np.array(value, dtype=p.dtype, ndmin=p.ndim)
                    setattr(el, name, va)
                elif isinstance(p, tuple):
                    va = tuple(value)
                    setattr(el, name, va)
                else:
                    setattr(el, name, value)

            elInputs = elStruct['input']
            if not elInputs:
                elInputs = []
            elif not isinstance(elInputs, list):
                elInputs = [elInputs]
            for ip in elInputs:
                el.addInput(ip['label'], ip['component'])

            self.elements.append(el)
            self.elementDict[elLabel] = el
            self.elementLabels.append(elLabel)

        # create targetDict
        self.targetDict = {l:[] for l in self.elementLabels}
        for el in self.elements:
            for il in el.inputElementLabels:
                self.targetDict[il].append(el.label)

        self.initialized = False


    def parametersFromStruct(self, simStruct, changeableOnly = False):
        self.deltaT = simStruct['deltaT']
        self.tZero = simStruct['tZero']
        self.initialized = False

        nElements = simStruct['nElements']
        elementsNotFound = []
        elementsOverwritten = {l: False for l in self.elementLabels}

        for i in range(nElements):
            elStruct = simStruct['elements'][i]
            elLabel = elStruct['label']

            if not elLabel in self.elementDict:
                elementsNotFound.append(i)
                continue
            el = self.elementDict[elLabel]
            if type(el).__name__ != elStruct['class']:
                elementsNotFound.append(i)
                continue

            pStruct = elStruct['param']
            for pName, pStatus in el.parameters.items():
                if pName in elStruct['param'] and \
                        (not changeableOnly or pStatus >= PS_CHANGEABLE):
                    value = elStruct['param'][pName]
                    p = getattr(el, pName)
                    if isinstance(p, np.ndarray):
                        va = np.array(value, dtype=p.dtype, ndmin=p.ndim)
                        setattr(el, pName, va)
                    else:
                        setattr(el, pName, value)
            elementsOverwritten[el.label] = True

        unchangedElements = [k for k, v in elementsOverwritten.items() if not v]

        if elementsNotFound or unchangedElements:
            msg = 'Warning: '
            if unchangedElements:
                msg += 'Some elements in the simulator object were not ' \
                        'specified in the parameter struct and will retain ' \
                        'their previous settings: '
                for label in unchangedElements:
                    el = self.elementDict[label]
                    msg += f'{type(el).__name__} element "{el.label}", '
                msg = msg[:-2] + '\n'
            if elementsNotFound:
                msg += 'For some elements specified in the parameter struct, ' \
                        'no matching elements were found in the simulator ' \
                        'object: '
                for i in elementsNotFound:
                    s = simStruct['elements'][i]
                    msg += f'{s["class"]} element "{s["label"]}", '
                msg = msg[:-2] + '\n'
            warning(msg)

    def toStruct(self):
        sElements = []
        for el in self.elements:
            si = []
            for i in range(el.nInputs):
                si.append({
                    'label': el.inputElementLabels[i],
                    'component': el.inputComponents[i]
                })
            sp = {}
            for p in el.parameters:
                v = getattr(el, p)
                if isinstance(v, np.ndarray):
                    sp[p] = v.tolist()
                else:
                    sp[p] = v

            se = {
                'label': el.label,
                'class': el.__class__.__name__,
                'param': sp,
                'nInputs': int(el.nInputs),
                'input': si
            }
            sElements.append(se)

        s = {
            'deltaT': self.deltaT,
            'tZero': self.tZero,
            'nElements': len(self.elements),
            'elementLabels': self.elementLabels,
            'elements': sElements
        }
        return {'simulator': s}

    def saveComponentsToMat(self, filename):
        ''' Save all element components to .mat file.

        When the resulting file is loaded in Matlab, it will contain a
        cell array of structs 'elements', with one struct for each element
        in the simulator. Each struct has a field 'label' containing
        the element label, and one field for each component of the
        element.

        Args:
            filename (str): Valid name for a .mat file.
        '''
        elements = []
        for el in self.elements:
            d = {'label': el.label}
            for c in el.components.keys():
                d[c] = getattr(el, c)
            elements.append(d)

        savemat(filename, {'elements': elements})



