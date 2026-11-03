using OptimizationBase, OptimizationJSOSolvers, ADTypes, ForwardDiff
using Test

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
rosenbrock_params = [1.0, 100.0]
optf = OptimizationFunction(rosenbrock, ADTypes.AutoForwardDiff())

@testset "TRON" begin
    @testset "inactive bounds" begin
        prob = OptimizationProblem(optf, zeros(2), rosenbrock_params; lb = [-1.0, -1.0], ub = [1.5, 1.5])
        sol = solve(prob, TRON())
        @test successful_retcode(sol)
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-6
    end

    @testset "active bounds" begin
        prob = OptimizationProblem(optf, zeros(2), rosenbrock_params; lb = [-1.0, -1.0], ub = [0.5, 0.5])
        sol = solve(prob, TRON())
        @test successful_retcode(sol)
        @test sol.u ≈ [0.5, 0.25] atol = 1.0e-6
    end

    @testset "MaxSense" begin
        negated = OptimizationFunction((x, p) -> -rosenbrock(x, p), ADTypes.AutoForwardDiff())
        prob = OptimizationProblem(negated, zeros(2), rosenbrock_params; sense = MaxSense)
        sol = solve(prob, TRON())
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-6
        @test sol.objective ≈ 0.0 atol = 1.0e-10
    end

    @testset "NoAD with a user grad and scaled hess" begin
        grad!(G, x, p) = ForwardDiff.gradient!(G, x -> rosenbrock(x, p), x)
        hess!(H, x, p) = (ForwardDiff.hessian!(H, x -> rosenbrock(x, p), x); H .*= 2)
        noad = OptimizationFunction(rosenbrock; grad = grad!, hess = hess!)
        prob = OptimizationProblem(noad, zeros(2), rosenbrock_params; lb = [-1.0, -1.0], ub = [0.5, 0.5])
        sol = solve(prob, TRON())
        @test successful_retcode(sol)
        @test sol.u ≈ [0.5, 0.25] atol = 1.0e-6
    end

    @testset "maxiters" begin
        prob = OptimizationProblem(optf, [-1.2, 1.0], rosenbrock_params)
        sol = solve(prob, TRON(); maxiters = 2)
        @test sol.retcode == ReturnCode.MaxIters
    end

    @testset "callback stops" begin
        prob = OptimizationProblem(optf, [-1.2, 1.0], rosenbrock_params)
        sol = solve(prob, TRON(); callback = (state, loss) -> state.iter >= 1)
        @test sol.retcode == ReturnCode.Terminated
    end

    @testset "non-Vector u0 and bounds" begin
        storage = [0.0, 0.0, -1.0, -1.0, 1.5, 1.5]
        prob = OptimizationProblem(optf, view(storage, 1:2), rosenbrock_params;
            lb = view(storage, 3:4), ub = view(storage, 5:6))
        sol = solve(prob, TRON())
        @test successful_retcode(sol)
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-6
    end
end
