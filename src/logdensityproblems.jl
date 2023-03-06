"""
    TemperedLogDensityProblem

A tempered log density function implementing the LogDensityProblem.jl interface.

# Fields
$(FIELDS)
"""
struct TemperedLogDensityProblem{L,T}
    "underlying log density; assumed to implement LogDensityProblems.jl interface"
    logdensity::L
    beta::T
end

LogDensityProblems.capabilities(::Type{<:TemperedLogDensityProblem{L}}) where {L} = LogDensityProblems.capabilities(L)
LogDensityProblems.dimension(tf::TemperedLogDensityProblem) = LogDensityProblems.dimension(tf.logdensity)
LogDensityProblems.logdensity(tf::TemperedLogDensityProblem, x) = tf.beta * LogDensityProblems.logdensity(tf.logdensity, x)
function LogDensityProblems.logdensity_and_gradient(tf::TemperedLogDensityProblem, x)
    y, ∇y = LogDensityProblems.logdensity_and_gradient(tf.logdensity, x)
    return tf.beta .* y, tf.beta .* ∇y
end

struct PathTemperedLogDensityProblem{L1,L2,T}
    "underlying log densities; assumed to implement LogDensityProblems.jl interface"
    logdensity1::L1
    logdensity2::L2
    beta::T
end

# TODO: how should these be combined?
LogDensityProblems.capabilities(::Type{<:PathTemperedLogDensityProblem{L1, L2}}) where {L1, L2} = LogDensityProblems.capabilities(L1)
LogDensityProblems.dimension(tf::PathTemperedLogDensityProblem) = LogDensityProblems.dimension(tf.logdensity1)
function LogDensityProblems.logdensity(tf::PathTemperedLogDensityProblem, x)
    tf.beta * LogDensityProblems.logdensity(tf.logdensity1, x) +
    (1 - tf.beta) * LogDensityProblems.logdensity(tf.logdensity2, x)
end
function LogDensityProblems.logdensity(tf::PathTemperedLogDensityProblem, x, signature::Vector{Bool})
    tf.beta * LogDensityProblems.logdensity(tf.logdensity1, x, signature) +
    (1 - tf.beta) * LogDensityProblems.logdensity(tf.logdensity2, x, signature)
end
function LogDensityProblems.logdensity_and_gradient(tf::PathTemperedLogDensityProblem, x)
    y1, ∇y1 = LogDensityProblems.logdensity_and_gradient(tf.logdensity1, x)
    y2, ∇y2 = LogDensityProblems.logdensity_and_gradient(tf.logdensity2, x)
    return tf.beta .* y1 + (1 - tf.beta) .* y2, tf.beta .* ∇y1 + (1 - tf.beta) .* ∇y2
end
function LogDensityProblems.logdensity_and_gradient(tf::PathTemperedLogDensityProblem, x, signature::Vector{Bool})
    y1, ∇y1 = LogDensityProblems.logdensity_and_gradient(tf.logdensity1, x, signature)
    y2, ∇y2 = LogDensityProblems.logdensity_and_gradient(tf.logdensity2, x, signature)
    return tf.beta .* y1 + (1 - tf.beta) .* y2, tf.beta .* ∇y1 + (1 - tf.beta) .* ∇y2
end
