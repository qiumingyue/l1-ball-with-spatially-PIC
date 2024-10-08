
#data generation
using Random, Distributions, Plots, NLsolve, LinearAlgebra, StatsPlots, MCMCChains, KernelDensity,DataFrames,CSV
seed = 135
include("l1-lattice-functions.jl")
include("l1-lattice-mainfunction.jl")
p = 30
p_non = 2
n = 500
true_beta = vcat(ones(p_non) .* 0.5, zeros(p-p_non))
rho = 0.5
phitau = 1
fun(t) = 0.2*t[1]^2

W = Matrix(CSV.read("matrix of data_tooth-53-54-55.csv",DataFrame,header=0))

exact_rate = 0.1

data = Data_generation(seed, n, W, true_beta, rho, fun, exact_rate, phitau)

#hyper-parameter set
bw_knots = 0.8
num_knots = 10
order = 3
niter = 10000

g = 1
a_eta = 0.1
b_eta = 0.1
a_phitau = 0.1
b_phitau = 0.1
b_w = 1
a_lam = 3
b_lam = 1
a_phitau = 0.1
b_phitau = 0.1
bw = 1e-3

result = est_spa(data)
