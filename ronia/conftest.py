"""Top-level configuration for PyTest

"""

import pytest

import ronia


@pytest.fixture(scope="module", params=[
    lambda formula, *args: lambda input_data, y: ronia.bayespy.GAM(
        # HACK: Let's use *args to allow feeding the `tau` parameter
        # only for `ronia.numpy.GAM`. Thus, we leave *args unused for
        # `ronia.bayespy.GAM`.
        formula
    ).fit(input_data, y),
    lambda formula, *args: lambda input_data, y: ronia.numpy.GAM(
        formula, *args
    ).fit(input_data, y)
])
def fit_model(request):
    return request.param
