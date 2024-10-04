
#data generation
using Random, Distributions, Plots, NLsolve, LinearAlgebra, StatsPlots, MCMCChains, KernelDensity, DataFrames,CSV, ROCCurves
seed = 55
rho = 0.5

include("l1-geo-functions.jl")
include("l1-geo-mainfunction.jl")
logR = Matrix(CSV.read("distance_tooth.csv",DataFrame,header=0))


exact_rate = 0.1

#hyper-parameter set
bw_knots = 0.8
num_knots = 10
order = 3
niter = 10000

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
prob = 0.95 #Credible Interval


p = 20
p_non = 2
n = 500
true_beta = vcat(ones(p_non) .* 0.5, zeros(p-p_non))
a_r = 2
b_r = 1
R_phi = exp.(-(a_r .* logR).^b_r)
phitau = 1
fun(t) = 0.5*t[1]
data = Data_generation(seed, n, R_phi, true_beta, rho, fun, exact_rate, phitau)
result = est_spa(data)
