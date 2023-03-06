"""
    make_tempered_model([sampler, ]model, beta)

Return an instance representing a `model` tempered with `beta`.

The return-type depends on its usage in [`compute_tempered_logdensities`](@ref).
"""
# TODO: Hacky fix
make_tempered_model(sampler::TemperedSampler, model, beta) = make_tempered_model(model, beta)
function make_tempered_model(model, beta)
    if !implements_logdensity(model)
        error("`make_tempered_model` is not implemented for $(typeof(model)); either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end

    return TemperedLogDensityProblem(model, beta)
end
function make_tempered_model(model::AbstractMCMC.LogDensityModel, beta)
    return AbstractMCMC.LogDensityModel(TemperedLogDensityProblem(model.logdensity, beta))
end
make_tempered_model(sampler, model, prior, beta) = make_tempered_model(model, prior, beta)
function make_tempered_model(model, prior, beta)
    if !implements_logdensity(model)
        error("`make_tempered_model` is not implemented for $(typeof(model)); either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end
    if !implements_logdensity(prior)
        error("`make_tempered_model` is not implemented for $(typeof(prior)); either implement explicitly, or implement the LogDensityProblems.jl interface for `prior`")
    end

    return PathTemperedLogDensityProblem(model, prior, beta)
end
function make_tempered_model(model::AbstractMCMC.LogDensityModel, prior::AbstractMCMC.LogDensityModel, beta)
    return AbstractMCMC.LogDensityModel(PathTemperedLogDensityProblem(model.logdensity, prior.logdensity, beta))
end

"""
    logdensity(model, x)

Return the log-density of `model` at `x`.
"""
function logdensity(model, x)
    if !implements_logdensity(model)
        error("`logdensity` is not implemented for `$(typeof(model))`; either implement explicitly, or implement the LogDensityProblems.jl interface for `model`")
    end
    return LogDensityProblems.logdensity(model, x)
end
logdensity(model::AbstractMCMC.LogDensityModel, x) = LogDensityProblems.logdensity(model.logdensity, x)
