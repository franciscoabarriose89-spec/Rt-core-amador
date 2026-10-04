# rt-core-amador

Lightweight mathematical tools implemented in pure Python. The package has no
runtime dependencies and requires Python 3.10 or newer.

> **Scope:** this is a new, self-contained implementation inspired by the
> requested project idea. "Amador Transform" here means the explicitly
> documented phase-modulated discrete Fourier transform below; it does not
> claim to reproduce an earlier implementation or an independently established
> mathematical definition.

## Install

```console
python -m pip install rt-core-amador
```

## Quick start

```python
from rt_core_amador import AmadorTensor, AmadorTransform

tensor = AmadorTensor([[1, 2], [3, 4]])
print((tensor * 2).to_list())

transform = AmadorTransform(phase=0.2)
coefficients = transform.forward([1, 0, -1, 0])
restored = transform.inverse(coefficients)
```

## API

### `AmadorTensor`

A small dense tensor backed by a flat tuple of Python floats. Construction
accepts a scalar or rectangular nested list/tuple. Supports element-wise
addition, subtraction, multiplication and division with another tensor of the
same shape or a scalar, plus `dot()` for vectors and `matmul()` for matrices.
This intentionally does not implement NumPy-style broadcasting.

### `AmadorTransform`

`forward(values)` computes a DFT normalized by `1/N` and applies a phase
rotation `exp(i * phase * k)` to frequency bin `k`. `inverse(coefficients)`
removes that rotation and computes the corresponding inverse. Both operations
accept a non-empty one-dimensional sequence; output values are Python complex
numbers.

### `NavierStokesOptimizer`

A simple explicit 2D incompressible Navier–Stokes projection step on a
rectangular periodic grid. Call `step(u, v, dt)` with same-sized 2D velocity
grids. It returns `(u_next, v_next, pressure)`. This is a basic educational
finite-difference solver, not a production CFD package: explicit time steps
must satisfy stability constraints, and boundary conditions are periodic.

### `KahlerManifold`

Numerically estimates the Hermitian metric
`g[i, j] = d²φ / dz[i] d(conjugate(z[j]))` from a real-valued potential
`φ(z)`. The default potential `sum(abs(z_i)**2)` yields the identity metric.
`metric_at(point)` returns a matrix of complex values and `norm_squared(point,
tangent)` evaluates the induced squared norm. Finite differences are
approximate and the supplied potential should be smooth and strictly
plurisubharmonic near the point.

## Development

Run the tests with:

```console
python -m unittest discover -s tests -v
```

Build the source and wheel distributions with:

```console
python -m pip install --upgrade build
python -m build
```

## PyPI publishing

The GitHub Actions workflow at
`.github/workflows/publish.yml` builds on a GitHub Release and publishes with
PyPI Trusted Publishing (OIDC). Configure a trusted publisher on PyPI for this
GitHub repository, workflow file `publish.yml`, and the `pypi` environment.
Create a GitHub environment named `pypi` before publishing. No PyPI API token
is stored in GitHub secrets.
