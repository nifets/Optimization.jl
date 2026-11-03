module OptimizationJSOSolvers

using OptimizationBase: OptimizationBase, OptimizationCache
using SciMLBase: SciMLBase
using OptimizationNLPModels: NLPModelsAdaptor, build_nlpmodel_meta
using NLPModels: NLPModels
using JSOSolvers: JSOSolvers, TronSolver
using SolverCore: SolverCore, GenericExecutionStats

export TRON

"""
    TRON(; max_cgiter = 50, cgtol = 0.1)

Trust-region Newton method for bound-constrained problems from
[JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl).

References

  - C.-J. Lin and J. J. Moré. Newton's Method for Large Bound-Constrained Optimization Problems (1999), SIAM Journal on Optimization, 9, 4, pp. 1100-1127.
"""
@kwdef struct TRON{T}
    max_cgiter::Int = 50
    cgtol::T = 0.1
end

SciMLBase.has_init(::TRON) = true
SciMLBase.allowscallback(::TRON) = true
SciMLBase.allowsbounds(::TRON) = true
SciMLBase.requiresgradient(::TRON) = true
SciMLBase.requireshessian(::TRON) = true

map_retcode(status::Symbol) = map_retcode(Val(status))
map_retcode(::Union{Val{:first_order}, Val{:small_step}}) = SciMLBase.ReturnCode.Success
map_retcode(::Val{:max_time}) = SciMLBase.ReturnCode.MaxTime
map_retcode(::Union{Val{:max_iter}, Val{:max_eval}}) = SciMLBase.ReturnCode.MaxIters
map_retcode(::Val{:user}) = SciMLBase.ReturnCode.Terminated
map_retcode(::Val) = SciMLBase.ReturnCode.Failure

function SciMLBase.__solve(cache::OptimizationCache{<:TRON})
    T = eltype(cache.u0)
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)

    meta = NLPModels.NLPModelMeta(build_nlpmodel_meta(cache); minimize = true)
    nlp = NLPModelsAdaptor(cache, meta, NLPModels.Counters())
    solver = TronSolver(nlp)
    stats = GenericExecutionStats(nlp)

    function callback(nlp, solver, stats)
        state = OptimizationBase.OptimizationState(;
            iter = stats.iter, u = solver.x, objective = stats.objective,
            grad = solver.gx, p = cache.p, original = stats
        )
        cache.callback(state, stats.objective) && SolverCore.set_status!(stats, :user)
        return nothing
    end

    SolverCore.solve!(
        solver, nlp, stats;
        x = Vector{T}(cache.u0),
        atol = T(something(cache.solver_args.abstol, √eps(T))),
        rtol = T(something(cache.solver_args.reltol, √eps(T))),
        max_iter = something(maxiters, typemax(Int)),
        max_time = Float64(something(maxtime, Inf)),
        max_cgiter = cache.opt.max_cgiter,
        cgtol = T(cache.opt.cgtol),
        callback
    )

    opt_stats = OptimizationBase.OptimizationStats(;
        iterations = stats.iter, time = stats.elapsed_time
    )

    return SciMLBase.build_solution(
        cache, cache.opt, stats.solution, stats.objective;
        original = stats, retcode = map_retcode(stats.status), stats = opt_stats
    )
end

end
