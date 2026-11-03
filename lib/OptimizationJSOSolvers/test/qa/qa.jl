using SciMLTesting, OptimizationJSOSolvers, JET
using Test

include(normpath(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "rendered_docs.jl")))

run_qa(
    OptimizationJSOSolvers;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (; ignore = ()),
        all_qualified_accesses_are_public = (; ignore = (:OptimizationState, :_check_and_convert_maxiters, :_check_and_convert_maxtime)),
    ),
)
