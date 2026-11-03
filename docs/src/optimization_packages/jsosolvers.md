# JSOSolvers.jl

`OptimizationJSOSolvers.jl` connects Optimization.jl problems to
[`JSOSolvers.jl`](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) from
JuliaSmoothOptimizers. It currently wraps TRON, a trust-region Newton method for
unconstrained and bound-constrained problems. TRON only needs Hessian-vector
products, and uses a user-supplied `hess` (e.g. a Gauss–Newton approximation) when
no `hv` is given.

## Installation: OptimizationJSOSolvers.jl

```julia
import Pkg
Pkg.add("OptimizationJSOSolvers")
```

## Methods

```@docs
OptimizationJSOSolvers.TRON
```

## Example

```julia
using OptimizationBase, OptimizationJSOSolvers, ADTypes

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
optf = OptimizationFunction(rosenbrock, ADTypes.AutoForwardDiff())
prob = OptimizationProblem(optf, zeros(2), [1.0, 100.0]; lb = [-1.0, -1.0], ub = [0.5, 0.5])

sol = solve(prob, TRON())
```
