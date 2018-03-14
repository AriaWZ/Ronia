# Ronia – Generalized additive models in Python with Bayesian twist

Generalized additive model is a predictive model which is defined
as a sum of terms which are calibrated using observation data. This
package provides a hopefully pleasant interface for configuring and fitting
such models. Bayesian interpretation of model parameters is promoted.


## Summary

Generalized additive models provide a surprisingly general framework for
building predictive models for both production software and research work.
This module provides tools for building the model terms as decompositions
of various basis functions. It is possible to model the terms as
Gaussian processes (with reduced dimensionality) of various kernels and
piecewise linear functions. Of course, very simple terms like lines and
constants are also supported (these are just very simple basis functions).

The uncertainty in the weight parameter distributions is modeled using
Bayesian statistic with the help of the superb package [BayesPy](http://www.bayespy.org/index.html).

The work is in very early stage, so many features are still missing.


## Examples

### Polynomial regression on 'roids

Start with very simple dataset

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ronia
from ronia.utils import pipe
from ronia.keyfunction import x


np.random.seed(42)


# Define dummy data
input_data = 10 * np.random.rand(30)
y = (
    5 * input_data +
    2.0 * input_data ** 2 +
    7 +
    np.random.randn(len(input_data))
)
```

The object `x` is a convenience tool for defining input data maps
as if they were just Numpy arrays (or Pandas DataFrames).

```python
# Define model
a = ronia.formulae.Scalar(prior=(0, 1e-6))
b = ronia.formulae.Scalar(prior=(0, 1e-6))
intercept = ronia.formulae.Scalar(prior=(0, 1e-6))
formula = a * x + b * x ** 2 + intercept
model = ronia.BayesianGAM(formula).fit(input_data, y)
```

The model attribute `model.theta` characterizes the Gaussian posterior distribution of the model parameters vector.

#### Predicting with model

```python
model.predict(input_data[:3])
# array([  99.25493083,   23.31063443,  226.70702106])
```

Predictions with uncertainty can be calculated as follows (`scale=2.0` roughly corresponds to the 95% confidence interval):

```python
model.predict_total_uncertainty(input_data[:3], scale=2.0)
# (array([ 97.3527439 ,  77.79515549,  59.88285762]),
#  array([ 2.18915289,  2.19725385,  2.18571614]))
```

#### Plotting results

```python
# Plot results
fig = ronia.plot.validation_plot(
    model,
    input_data,
    y,
    grid_limits=[0, 10],
    input_maps=[x, x, x],
    titles=["a", "b", "intercept"]
)
```

The grey band in the top figure is two times
the prediction standard deviation and, in the partial residual plots, two times
the respective marginal posterior standard deviation.

![alt text](./doc/source/images/example0-0.png "Validation plot")

It is also possible to plot the estimated Γ-distribution of the noise precision
(inverse variance) as well as the 1-D Normal distributions of each individual
model parameter.

```python
# Plot parameter probability density functions
fig = ronia.plot.gaussian1d_density_plot(model, grid_limits=[-1, 3])
```
![alt text](./doc/source/images/example0-1.png "1-D density plot")

#### Saving model on hard disk for later use (HDF5)

Saving

```python
model.save("/home/foobar/test.hdf5")
```

Loading

```python
model = BayesianGAM(formula).load("/home/foobar/test.hdf5")
```

### Gaussian process regression ("kriging")

```python
# Create some data
n = 50
input_data = np.vstack(
    (
        2 * np.pi * np.random.rand(n),
        np.random.rand(n),
    )
).T
y = (
    np.abs(np.cos(input_data[:, 0])) * input_data[:, 1] +
    1 +
    0.1 * np.random.randn(n)
)


# Define model
a = ronia.formulae.ExpSineSquared1d(
    np.arange(0, 2 * np.pi, 0.1),
    l=1.0,
    sigma=1.0,
    period=2 * np.pi,
    energy=0.99
)
intercept = ronia.formulae.Scalar(prior=(0, 1e-6))
formula = a(x[:, 0]) * x[:, 1] + intercept
model = ronia.BayesianGAM(formula).fit(input_data, y)


# Plot results
fig = ronia.plot.validation_plot(
    model,
    input_data,
    y,
    grid_limits=[[0, 2 * np.pi], [0, 1]],
    input_maps=[x[:, 0:2], x[:, 1]],
    titles=["a", "intercept"]
)


# Plot parameter probability density functions
fig = ronia.plot.gaussian1d_density_plot(model, grid_limits=[-1, 3])
```

![alt text](./doc/source/images/example1-0.png "Validation plot")
![alt text](./doc/source/images/example1-1.png "1-D density plot")

## To-be-added features

- **TODO** Multivariate basis functions
- **TODO** Multi-dimensional observations
- **TODO** Hyperpriors for model parameters
- **TODO** Dynamically changing models
