# Cosivina for Python - Compose, Simulate, and Visualize Neurodynamic Architectures
## An open source module for Python

Written by Sebastian Schneegans

Copyright (c) 2021 Sebastian Schneegans <Sebastian@Schneegans.de>  
Published under the Simplified BSD License (see LICENSE_BSD.txt)


## Overview

Cosivina is a free object-oriented framework to construct dynamic field architectures and simulate the evolution of activation in these architectures. For more information on Dynamic Field Theory in general, please visit [www.dynamicfieldtheory.org](http://www.dynamicfieldtheory.org).

Cosivina was originally written for Matlab. This Python implementation offers the core functionality of the Matlab version, but currently lacks graphical user interfaces, only contains a subset of architecture elements and currently only supports one- or two-dimensional neural fields.


## Quick start

- Clone the repository or download the code as zipfile.
- Open your preferred Python IDE, open one of the example files from the base folder (e.g. `exampleCoupling`, and run it.


## Simulation performance and just-in-time compilation

Cosivina for Python can be operated in two modes: as regular Python code, or with just-in-time compilation using the [Numba](https://numba.pydata.org/) package. The no-numba version has performance that is typically somewhat slower (and sometimes a lot slower) than running the equivalent code in Matlab. The numba version compiles each class and function when it is used for the first time. This can take quite long, potentially up to several minutes for code that uses many different classes. However, once compiled, the execution of subsequent function calls is faster than in the no-numba version. Both version are based on the scientific computing package numpy, and all element components are numpy arrays.

One key performance advantage of the numba version is that it significantly reduces the overhead of calling methods and accessing elements of custom classes. Both in Matlab and standard Python, calling a method of a custom class (such as any architecture element in cosivina) takes extra time compared to executing the content of the method outside of a class. In dynamic field architectures in which individual steps are not very computation-intensive (such as in most architectures composed only of one-dimensional fields), this overhead can make up a substantial part of the total computation time. The compiled code largely avoids this overhead, and should come close to optimized procedural Matlab code.

Some other computations tend to be slower than in Matlab. This unfortunately includes the convolution operation (especially the 2D convolution), which does not yet have a full numba implementation. However, the convolution using the FFT method is implemented very efficiently in numpy, and should be preferred for all 2D convolutions (if you have architectures in Matlab that you would like to convert to use FFT kernels, you can do so with the recently added auxiliary Matlab function `switchToFFT`).

The example files (`exampleThreeLayerField`, `exampleCoupling`, and `exampleSceneRepresentation`) each come in a Python (`.py`) and a Matlab (`.m`) version. Both run the same simulations and measure the computation time, so you can test how they perform on your system.


## Choosing and switching between numba and no-numba mode

You can choose which version of cosivina is used by importing either `cosivina.numba` or `cosivina.nonumba` (you should never import just `cosivina`). In the example scripts, both versions are prepared, and you can choose by commenting out one of them (with the hash character), either

    from cosivina.nonumba import *
    \# from cosivina.numba import *

or

	\# from cosivina.nonumba import *
    from cosivina.numba import *

Switching between versions once you have imported one of them will require restarting the Python kernel (similar to a `clear all` in Matlab). In the Spyder IDE, you can do this by choosing in the menu bar `Consoles`, then `Restart kernel`. In Pyzo, it is `Shell`, `Restart`.

Note that the use of `from module import *` is somewhat frowned upon in Python, but allows the code to be most similar to Matlab. A cleaner way would be use e.g. `import cosivina.numba as cv`, and then calling all class constructors from the `cv` module, e.g. `sim = cv.Simulator()` and `sim.addElement(cv.NeuralField(...))`.


## Compatibility with Matlab version

The code to create and run an architecture is extremely similar between Matlab and Python, with only a few small differences described in the following sections. In addition, parameter files in json format are interchangeable between the two languages, meaning you can e.g. create an architecture in Matlab, and then load it in Python.

It is also possible to save all element components (such as dynamic field activations) of a simulator in Python using the `Simulator.saveComponentsToMat` method, and load it in Matlab for analysis. Note that this will not create a `Simulator` object that can be run in Matlab; it rather creates a list of structs, one for each element, with the element's components saved as fields of those structs. Directly converting a `Simulator` object between Matlab and Python is currently not possible.


## Getting started for Matlab users

A popular way to work with Python is to install an environment like [Anaconda](https://www.anaconda.com/products/individual), which provides a Python interpreter and an integrated development environment (IDE), such as Spyder, that works similar to Matlab. A somewhat more light-weight alternative to Anaconda for Windows is [Winpython](https://winpython.github.io/) with the Pyzo IDE. 

Once an environment is installed (and cosvina downloaded), you can load one of the cosivina example files and run it as you would in Matlab.


## Basic commands and differences to Matlab

Create a new simulator object as

    sim = Simulator()

You don't need a semicolon after a command to suprres output. Optional arguments can be given by their name in Python, e.g.

    sim = Simulator(file='presetCouplingFFT.json')

Add elements with

    sim.addElement(NeuralField('field u', (1, 100), 10, -5, 4))

Note that all size parameters must be given as a two-element tuple `(rows, columns)` in parenthes / round brackets. You can also define it as a variable beforehand, e.g. as

    fieldSize = (1, 100)

For lists of element labels, use brackets rather than curly braces as in Matlab (the resulting container is simply called a list, and replaces the Matlab cell array):

    sim.addElement(NeuralField('field u', fieldSize), ['stim 1', 'stim 2'], ['output', 'output'])
	
For boolean variables (e.g. the `circular` parameter in a kernel), use values `True` or `False` (capitalized):

    sim.addElement(GaussKernel1D('u -> u', fieldSize, 5, 10, True, True), 'field u', 'output', 'field u')

A simulation can be run just as in Matlab:

	sim.run(tMax)

You can plot components using the `matplotlib` module, which provides function that are very similar to Matlab, e.g.

    import matplotlib.pyplot as plt
    plt.plot(sim.getComponent('field s', 'activation')[0], 'b')

or for 2D fields

    plt.imshow(sim.getComponent('field v', 'activation'))

Note that for plotting the 1D field, the component is indexed with `[0]`. This is because all components are 2D arrays, which are realized in Python as vectors of vectors. The plot functions expects a single vector, so we extract the first vector from the matrix (and unlike Matlab, Python uses 0-based indices, so the first row of the matrix has index 0).

You can save all components in the simulator to a .mat file (which can then be loaded in Matlab) using

    sim.saveComponentsToMat('results.mat')

This uses the function `savemat` from the scipy.io module.


## Implemented elements

Currently only a subset of all elements from the Matlab version are implemented. These generally behave exactly like their Matlab counterparts, and take the same parameters. To get a description of an element class and all its methods, type e.g.

    help(NeuralField)

in the Python console (you need to have cosivina imported already for this to work). To see only the constructor call, use

    help(NeuralField.__init__)

(with two underscores on each side of `init`). Similarly, get help on the Simulator class and its methods using e.g.

help(Simulator)
help(Simulator.saveComponentsToMat)

The following elements are currently implemented:

- `NeuralField` (with no more than two dimensions)
- `GaussStimulus1D` and `GaussStimulus2D`
- `BoostStimulus`
- `GaussKernel1D` and `GaussKernel2D`
- `MexicanHatKernel1D and `MexicanHatKernel2D`
- `LateralInteractions1D` and `LateralInteractions2D`
- `KernelFFT` (with no more than two dimensions)
- `SumInputs`
- `ScaleInput`
- `Transpose`
- `ExpandDimension2D` (shouldn't really be needed, use `Transpose` instead)
- `SumDimension` (note: `sumDimensions` argument must be a numpy array)
- `SumAllDimensions`
- `NormalNoise` (with no more than two dimensions)


## Example files

Three example files are included to illustrate different aspects of the module:

- `exampleThreeLayerField`: Create the three-layer field architecture from the Matlab examples, load settings from a json file, run it and measure computation time, and then run it with (manual) online plotting. Note that online plotting may be very slow in some environments; in this case, you should comment out the last section of the code). Also note that you could load the whole architecture from the json file, re-creating it in Python is just for illustration.

- `exampleCoupling`: Creates the coupling architecture from the Matlab examples, with two one-dimensional field interacting with a single two-dimensional field along different dimensions. Runs the model with timing, and then plots the final state.

- `exampleSceneRepresentation`: Loads the scene representation architecture from a json file and runs it with timing, first in the default version, then with all 2D interactions converted to KernelFFT elements (code for the conversion is in the Matlab version of the example). Finally, the Python version saves all element components in the architecture to a `.mat` file, and the Matlab version loads that file and compares field activations between the two implementations (they should be identical except for minimal numerical deviations on the order of 10^-15).






