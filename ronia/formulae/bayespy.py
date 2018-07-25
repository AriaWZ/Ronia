import attr
import bayespy as bp
import numpy as np
import scipy as sp
from scipy import interpolate

from ronia import utils
from ronia.utils import listmap, rlift_basis


def concat_gaussians(gaussians: list):
    # gaussians = [(μ1, Λ1), (μ2, Λ2)]
    return (
        np.hstack([g[0] for g in gaussians]),
        sp.linalg.block_diag(*[g[1] for g in gaussians])
    )


def design_matrix(input_data, basis):
    return np.hstack([
        f(input_data).reshape(-1, 1) for f in basis
    ])


@attr.s(frozen=True)
class BayesPyFormula():
    """BayesianGAM model configuration settings

    Parameters
    ----------
    bases : list
        Each element is a list of basis functions and corresponds to a term
        in the additive model formula
    prior : tuple
        Mean and precision matrix of the Gaussian prior distribution

    Example
    -------
    Formulas can be summed up

    """

    bases = attr.ib()
    prior = attr.ib()

    def __add__(self, other):
        return BayesPyFormula(
            bases=self.bases + other.bases,
            prior=concat_gaussians([self.prior, other.prior])
        )

    def __mul__(self, input_map):
        # What other linear operations should be supported?
        return BayesPyFormula(
            bases=[
                listmap(
                    lambda f: lambda t: f(t) * input_map(t)
                )(basis) for basis in self.bases
            ],
            prior=self.prior
        )

    def __len__(self):
        return len(self.bases)

    def __call__(self, *input_maps):
        # TODO: Transform basis
        return BayesPyFormula(
            bases=[
                rlift_basis(f, m) for (f, m) in zip(self.bases, input_maps)
            ],
            prior=self.prior
        )

    def build_theta(self):
        return bp.nodes.Gaussian(self.prior[0], self.prior[1])

    def build_Xi(self, input_data, i):
        return design_matrix(input_data, self.bases[i])

    def build_Xs(self, input_data):
        return [
            self.build_Xi(input_data, i) for i, _ in enumerate(self.bases)
        ]

    def build_X(self, input_data):
        return np.hstack(self.build_Xs(input_data))

    def build_F(self, input_data, theta=None):
        """Constructs the forward model node

        """
        return bp.nodes.SumMultiply(
            "i,i",
            self.build_theta() if theta is None else theta,
            self.build_X(input_data)
        )


#
# Operations between formulae
#


def Flatten(formula, prior=None):
    """Flatten formula

    Bases: [[f1, f2], [g1, g2, g3]] => [[f1, f2, g1, g2, g3]]

    """
    return BayesPyFormula(
        bases=[utils.flatten(formula.bases)],
        prior=formula.prior if prior is None else prior
    )


def Sum(formulae, prior=None):
    """Sum (i.e. concatenate) formulae

    Bases: ([[f1, f2], [g1, g2]], [[h1]]) => [[f1, f2], [g1, g2], [h1]]

    """
    priors = [formula.prior for formula in formulae]
    return BayesPyFormula(
        bases=[utils.flatten([formula.bases for formula in formulae])],
        prior=concat_gaussians(priors) if prior is None else prior
    )


def Kron(a, b):
    """Take the tensor product of two BayesPyFormula bases

    Non-commutative!

    Parameters
    ----------
    a : BayesPyFormula
    b : BayesPyFormula

    Returns
    -------
    BayesPyFormula

    Let ``u, v`` be eigenvectors of matrices ``A, B``, respectively. Then
    ``u ⊗ v`` is an eigenvector of ``A ⊗ B`` and ``λμ`` is the corresponding
    eigenvalue.

    """
    # NOTE: This is somewhat experimental. The bases must correspond to
    #       "zero-mean" r.v.. Then Kronecker product of covariances
    #       corresponds to the product r.v. of independent r.v.'s.
    #       Check the formula of variance of product of independent r.v.'s.

    # In the same order as in a Kronecker product
    gen = (
        (f, g) for g in utils.flatten(b.bases) for f in utils.flatten(a.bases)
    )

    # Outer product of bases
    basis = listmap(
        lambda funcs: lambda t: funcs[0](t) * funcs[1](t)
    )(gen)

    # Kronecker product of prior means and covariances
    return BayesPyFormula(
        bases=[basis],
        prior=(
            np.kron(a.prior[0], b.prior[0]),
            np.kron(a.prior[1], b.prior[1])
        )
    )


#
# Custom formulae collection
#


def ExpSquared1d(
    grid, l, sigma, prior=None, mu_basis=None, mu_hyper=None, energy=0.99
):
    """Squared exponential model term

    Example
    -------

    .. code-block:: python

        formula = ExpSquared1d(
            grid=np.arange(-25, 35, 1.0),
            l=5.0,
            sigma=1.0,
            mu_basis=[lambda t: t],
            mu_hyper=(0, 1e-6)
        )

    """
    mu_basis = [] if mu_basis is None else mu_basis
    basis = utils.interp1d_1darrays(
        utils.scaled_principal_eigvecsh(
            utils.exp_squared(
                X1=grid.reshape(-1, 1),
                X2=grid.reshape(-1, 1),
                l=l,
                sigma=sigma
            ),
            energy=energy
        ),
        grid=grid,
        fill_value="extrapolate"
    )
    # Default prior is white noise
    prior = (
        (np.zeros(len(basis)), np.identity(len(basis)))
        if prior is None else prior
    )
    return BayesPyFormula(
        bases=[mu_basis + basis],
        prior=prior if mu_hyper is None else concat_gaussians(
            [mu_hyper, prior]
        )
    )


def ExpSineSquared1d(
    grid, l, sigma, period,
    prior=None, mu_basis=None, mu_hyper=None, energy=0.99
):
    mu_basis = [] if mu_basis is None else mu_basis
    basis = utils.interp1d_1darrays(
        utils.scaled_principal_eigvecsh(
            utils.exp_sine_squared(
                X1=grid.reshape(-1, 1),
                X2=grid.reshape(-1, 1),
                l=l,
                sigma=sigma,
                period=period
            ),
            energy=energy
        ),
        grid=grid,
        fill_value="extrapolate"
    )
    # Default prior is white noise
    prior = (
        (np.zeros(len(basis)), np.identity(len(basis)))
        if prior is None else prior
    )
    return BayesPyFormula(
        bases=[mu_basis + basis],
        prior=prior if mu_hyper is None else concat_gaussians(
            [mu_hyper, prior]
        )
    )


def WhiteNoise1d(
    grid, sigma, prior=None, mu_basis=None, mu_hyper=None, energy=1.0
):
    mu_basis = [] if mu_basis is None else mu_basis
    basis = utils.interp1d_1darrays(
        utils.scaled_principal_eigvecsh(
            utils.white_noise(n_dims=len(grid), sigma=sigma),
            energy=energy
        ),
        grid=grid,
        fill_value="extrapolate"
    )
    # Default prior is white noise
    prior = (
        (np.zeros(len(basis)), np.identity(len(basis)))
        if prior is None else prior
    )
    return BayesPyFormula(
        bases=[mu_basis + basis],
        prior=prior if mu_hyper is None else concat_gaussians(
            [mu_hyper, prior]
        )
    )


def Scalar(prior):
    basis = [lambda t: np.ones(len(t))]
    return BayesPyFormula(bases=[basis], prior=prior)


def Line(prior):
    basis = [lambda t: t]
    return BayesPyFormula(bases=[basis], prior=prior)


def Function(function, prior):
    basis = [function]
    return BayesPyFormula(bases=[basis], prior=prior)


def ReLU(grid, prior=None):
    relus = listmap(lambda c: lambda t: (t > c) * (c - t))(grid[1:-1])
    prior = (
        (np.zeros(len(grid) - 2), np.identity(len(grid) - 2))
        if not prior else prior
    )
    return BayesPyFormula(bases=[relus], prior=prior)


def FlippedReLU(grid, prior=None):
    relus = listmap(lambda c: lambda t: (t < c) * (c - t))(grid[1:-1])
    prior = (
        (np.zeros(len(grid) - 2), np.identity(len(grid) - 2))
        if not prior else prior
    )
    return BayesPyFormula(bases=[relus], prior=prior)


def TanH():
    raise NotImplementedError


def Gaussian1d():
    raise NotImplementedError


def BSpline1d(grid, order=3, extrapolate=True,
              prior=None, mu_basis=None, mu_hyper=None):
    """B-spline basis on a fixed grid

    Parameters
    ----------

    order : int
        Order of the spline function. Polynomial degree is ``order - 1``
    extrapolate : bool
        Extrapolate outside of the grid using basis functions "touching" the
        endpoints

    Number of spline basis functions is always ``N = len(grid) + order - 2``

    """
    # TODO: Verify that this doesn't break when scaling the grid
    #       (extrapolation + damping)

    mu_basis = [] if mu_basis is None else mu_basis
    grid_ext = utils.extend_spline_grid(grid, order)

    def build_basis_element(spline_arg):

        (knots, extrapolate, loc) = spline_arg

        def right_damp(t):
            return t > knots[-1]

        def left_damp(t):
            return knots[0] > t

        def element(t):
            sp_element = interpolate.BSpline.basis_element(
                knots,
                extrapolate if loc in (-1, 1) else False
            )
            return sp_element(t) if loc == 0 else (
                sp_element(t) * right_damp(t) if loc == -1 else
                sp_element(t) * left_damp(t)
            )

        return utils.compose2(np.nan_to_num, element)

    basis = listmap(build_basis_element)(
        utils.gen_spline_args_from_grid_ext(grid_ext, order, extrapolate)
    )

    # Default prior is white noise
    prior = (
        (np.zeros(len(basis)), np.identity(len(basis)))
        if prior is None else prior
    )
    return BayesPyFormula(
        bases=[mu_basis + basis],
        prior=prior if mu_hyper is None else concat_gaussians(
            [mu_hyper, prior]
        )
    )
