module Utils

using LinearAlgebra

const sigma2gamma = sigma -> 1 ./ (2 .* sigma .^ 2)
const gamma2sigma = gamma -> sqrt.(1 ./ (2 .* gamma))

export sigma2gamma, gamma2sigma

normalized(K::Function) = (x, y, args...) -> K(x, y, args...) / sqrt(K(x, x, args...) * K(y, y, args...))

kernel_asin(x, y, σ=1.0) = 2/π * asin((1 + x⋅y)/sqrt((1/(2σ^2) + 1 + x⋅x) * (1/(2σ^2) + 1 + y⋅y)))
const kernel_asin_normalized = normalized(kernel_asin)

kernel_acos(x, y, σ=1.0; n::Integer=1) = let
    fun = if n == 0
        kernel_acos_0
    elseif n == 1
        kernel_acos_1
    elseif n == 2
        kernel_acos_2
    else
        error("n must be 0, 1 or 2")
    end
    fun(x*√(σ), y*√(σ))
end
const kernel_acos_normalized = normalized(kernel_acos)

safe_acos(x) =
if x > 1.5
    @warn "acos($x) is out of range"
elseif x < -1.5
    @warn "acos($x) is out of range"
elseif x > 1
    0
elseif x < -1
    π
else
    acos(x)
end

theta(x, y) = safe_acos((x⋅y)/sqrt((x⋅x) * (y⋅y)))

# 1 - 1/π*acos((x ⋅ y)/(||x||*||y||)) = 1 - θ/π
kernel_acos_0(x, y) = 1 - theta(x, y)/π

# 1/π*||x||*||y||*(sin(θ) + (π - θ)*cos(θ))
kernel_acos_1(x, y) = let θ = theta(x, y)
    1/π * √(x⋅x) * √(y⋅y) * (sin(θ) + (π - θ)*cos(θ))
end

# 1/π*||x||^2*||y||^2*(3*sin(θ)*cos(θ) + (π - θ)*(1 + 2*cos^2(θ)))
kernel_acos_2(x, y) = let θ = theta(x, y)
    1/π * (x⋅x) * (y⋅y) * (3*sin(θ)*cos(θ) + (π - θ)*(1 + 2*cos(θ)^2))
end

end
