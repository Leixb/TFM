module Utils

sigma2gamma = sigma -> 1 ./ (2 .* sigma .^ 2)
gamma2sigma = gamma -> sqrt.(1 ./ (2 .* gamma))

export sigma2gamma, gamma2sigma

end
