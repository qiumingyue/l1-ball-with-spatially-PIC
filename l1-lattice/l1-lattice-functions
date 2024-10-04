
function Data_generation(seed, n, W, true_beta, rho, fun1, exact_rate, tau)
    Random.seed!(seed)
    N = size(W)[1]
    p = length(true_beta)
    Sigma = [rho^(abs(i-j)) for i in 1:p,j in 1:p]
    x = rand(MvNormal(Sigma),n)'
    locations = collect(1:N)
    area = rand(DiscreteUniform(1,N),n)
    TransMatrix = hcat(diagm(ones(N-1)),repeat([-1.0],N-1))
    A = [(i == j ? sum(W[:,i]) : 0) for i=1:N,j=1:N]
    W_a = copy(W .- diagm(ones(N)).*0.1)
    Sigmamatrix = Hermitian(inv(TransMatrix * Matrix(A.-W_a) * TransMatrix')./phitau)
    phicof = rand(MvNormal(Sigmamatrix),1)
    phicof = vcat(phicof,-sum(phicof))

    spaind = zeros(n,N) 
    for i in 1:N
        spaind[area .== locations[i],i] .= 1
    end
    linPred = exp.(x * true_beta .+ spaind * phicof)

    obv_point = 1 .+ rand(Poisson(2),n)
    u1 = rand(Uniform(0,1),n)
    tt = -log.(1 .- u1) ./ linPred
    function f!(F, t)
      F[1] = fun1(t) - tt[1]
    end
    trueTimes = nlsolve(f!, [1.0]).zero
    global  k = 2
    for i = 2:n
      global num = tt[k]
      function g!(F, x)
        F[1] = fun1(x) - num
      end
      trueTimes = vcat(trueTimes,nlsolve(g!, [1.0]).zero)
      k = k + 1
    end

    l = zeros(n)
    u = zeros(n)
    delta = ones(n) * 2
    ex_ind = unique(sort(rand(DiscreteUniform(1, n),Int(round(n*exact_rate,digits = 0)))))
    
    for i = 1:n
        if (i in ex_ind)
            l[i] = trueTimes[i]
            u[i] = trueTimes[i]
            delta[i] = 0
        else
            time_lag = rand(Exponential(1),obv_point[i])
            time_seq = vcat(0,cumsum(time_lag),Inf)
            for j in 2:length(time_seq)
                if time_seq[j-1] < trueTimes[i] <= time_seq[j]
                    l[i] = time_seq[j-1]
                    u[i] = time_seq[j]
                end
            end
        end
    end

    delta[u .== Inf] .= 3
    delta[l .== 0] .= 1
    data = hcat(l,u,x,area,delta,spaind * phicof)
    return(data)
end
function Ispline(x, order, knots)
    k = order + 1  
    m = length(knots)  
    n = m - 2 + k    
    t = vcat(repeat([1],k) * knots[1], knots[2:(m - 1)], repeat([1],k) * knots[m]) 
    yy_1 = begin
        local yy1 = zeros(n + k - 1, length(x))
        for l = k:n
            yy1[l,:] = (t[l] .<=  x .< t[l + 1])/(t[l + 1] - t[l])
        end
        yy1
    end
    yytem_1 = begin
        local yytem1 = yy_1
        for ii = 1:order
            yytem1 = begin
                local yytem2 = zeros(n + k - 1 - ii, length(x))
                for i = (k - ii):n
                    yytem2[i,:] = (ii + 1) .* ((x .- t[i]) .* yytem1[i, :] + (t[i + ii + 1] .- x) .* yytem1[i + 1,: ])./(t[i + ii + 1] - t[i])./ii
                end
                yytem2
            end
        end
        yytem1
    end
    index = begin
        local index_1 = zeros(length(x))
        for i = 1:length(x) 
            index_1[i] = sum(t .<= x[i])
        end
        index_1
    end 
    if order == 1 
        yy2 = begin 
            local yy = zeros(n - 1, length(x))
            for i = 2:n
                yy[i - 1, :] = (i .< index .- order .+ 1) .+ (i == index) .* (t[i + order + 1] - t[i]) .* yytem_1[i, :]./(order + 1)
            end
            yy
        end
    else
        yy2 = begin
            local yy = zeros(n - 1, length(x))
            for j = 1:length(x)
                for i = 2:n
                    if i < (index[j] - order + 1)
                        yy[i - 1, j] = 1
                    elseif index[j] >= i >= (index[j] - order + 1)
                        yy[i - 1, j] = (t[(i + order + 1):Int(index[j] + order + 1)] - t[i:Int(index[j])])' * yytem_1[i:Int(index[j]), j]/(order + 1)
                    else
                        yy[i - 1, j] = 0
                    end
                end
            end
            yy
        end
    end
    return(yy2)
end
function Mspline(x, order, knots)
    k = order 
    m = length(knots)  
    n = m - 2 + k    
    t = vcat(repeat([1],k) * knots[1], knots[2:(m - 1)], repeat([1],k) * knots[m]) 
    yy_1 = begin
        local yy1 = zeros(n + k - 1, length(x)) 
        for l = k:n 
            yy1[l,:] = (t[l] .<=  x .< t[l + 1])/(t[l + 1] - t[l])
        end
        yy1
    end
    if order == 1
        yytem_1 = yy1
    else
        yytem_1 = begin
            local yytem1 = yy_1
            for ii = 1:(order-1)
                yytem1 = begin
                    local yytem2 = zeros(n + k - 1 - ii, length(x))
                    for i = (k - ii):n
                        yytem2[i,:] = (ii + 1) .* ((x .- t[i]) .* yytem1[i, :] + (t[i + ii + 1] .- x) .* yytem1[i + 1,: ])./(t[i + ii + 1] - t[i])./ii
                    end
                    yytem2
                end
            end
            yytem1
        end
    end
    return yytem_1
end
function HMC_new_sampler(theta, epsilon, accept_rate, likefun, gradfun, Hessian; args...)
    #p_theta = length(theta)
    partical_ini = gradfun(theta; args...)
    Hessian_theta = Hermitian(inv(Hessian(theta;args...)))[:,:]
    delta_theta = rand(MvNormal(epsilon .* Hessian_theta))
    theta_prop = (theta .+ Hessian_theta * ((epsilon/2) .* partical_ini) .+ delta_theta)[:,1]

    #partical_prop = gradfun(theta_prop; args...)
    Hessian_prop = Hermitian(inv(Hessian(theta_prop;args...)))[:,:]

    logprop_ini = logpdf(MvNormal(theta[:,1],epsilon.*Hessian_theta), theta_prop)
    logini_prop = logpdf(MvNormal(theta_prop,epsilon.*Hessian_prop), theta)

    log_ini = likefun(theta; args...)
    log_prop = likefun(theta_prop; args...)

    logR = log_prop .- log_ini .- logprop_ini .+ logini_prop 


    if (rand(Uniform(0,1),1)[1] < exp.(logR)[1])
        theta = theta_prop
        accept_rate = accept_rate + 1
    end
    res = Dict("theta" => theta, "accept_rate" => accept_rate)
    return(res)
end
function log_Posterior_theta(beta; xcov, z, te2, spatialcof, lamtau, sigma2)
    sum1 = z' * xcov * beta
    sum2 = te2' * exp.(xcov * beta .+ spatialcof)
    sum3 = beta' * lamtau * beta ./(2 * sigma2)
    result = sum1 .-sum2 .-sum3
    return result
end
function partial_theta(beta; xcov, z, te2, spatialcof, lamtau, sigma2)
    sum1 = xcov' * z
    sum2 = xcov' * (te2 .* exp.(xcov * beta .+ spatialcof))
    sum3 = lamtau * beta ./ sigma2
    result = sum1 - sum2 - sum3
    return result
end
function Hessian_theta(beta; xcov, z, te2, spatialcof, lamtau, sigma2)
    hessian1 = xcov' * (xcov .* (te2 .* exp.(xcov * beta + spatialcof)))
    hessian2 = lamtau ./ sigma2
    result = hessian1 + hessian2
    return(result)
end
function tau_sampler(theta, lam, sigma2, Non_indicator, p)
    tau = zeros(p)
    num_non0 = sum(Non_indicator)
    if num_non0 > 0
        tem = lam[Non_indicator .==1] .* sqrt(sigma2) ./abs.(theta[Non_indicator .==1])
        tau[Non_indicator .== 1] = min.(1 ./vcat(rand.(InverseGaussian.(tem,1),1)...),1e6)
    end
    if num_non0 < p 
        tau[Non_indicator .== 0] .= rand(Gamma(1/2,2),p-num_non0)
    end
    return tau
end

function log_tau_density(tau, lam, theta, sigma2)
    res = -log.(tau)./2 .- theta.^2 ./(lam.^2 .*2 .*sigma2 .* tau) .- tau./2
    return res
end

function Nonzeros_sampler(tau, mu, p, lam, theta, sigma2, Non_indicator, tau_0)
    p_Non = -mu./lam./sqrt(sigma2)
    accept = 0
    p_Zero = log.(1 .- exp.(p_Non))

    w1 = p_Zero .+ log_tau_density(tau_0, lam, theta, sigma2) .+ rand(Normal(0,1),p) # .+ rand(GeneralizedExtremeValue(0,1,0),p)#randn(p) 
    w2 = p_Non .+ log_tau_density(tau, lam, theta, sigma2).+  rand(Normal(0,1),p)# .+ rand(GeneralizedExtremeValue(0,1,0),p)#randn(p)

    Non_indicator_prop = Int.(w1 .< w2)
    accept_rate = sum(Non_indicator_prop)/sum(Non_indicator)
    if rand(1)[1] < accept_rate
        Non_indicator = Non_indicator_prop
        accept = 1
    end
    return Dict("Non_indicator" => Non_indicator, "accept_rate" => accept)
end

function log_w_density(w, mu, Non_indicator, sigma2, lam, b_w)
    p_Non = -mu./lam./sqrt(sigma2)
    p_0 = log.(1 .- exp.(p_Non))
    p = length(Non_indicator)
    return p_Non' * Non_indicator + p_0' * (1 .- Non_indicator) + (p^b_w - 1) * log(1-w)
end
function wmu_sampler(w, Non_indicator, sigma2, lam, a_lam, b_lam, b_w, eps_w, accept)
    forward_lb = max(w-eps_w, 0)
    forward_ub = min(w+eps_w, 1)
    forward_density = -log(forward_ub - forward_lb)

    mu = (w^(-1/a_lam)-1) * b_lam * sqrt(sigma2)

    w_new = rand(Uniform(forward_lb, forward_ub),1)[1]
    backward_lb = max(w_new-eps_w, 0)
    backward_ub = min(w_new+eps_w, 1)
    backward_density = -log(backward_ub - backward_lb)

    mu_new = (w_new^(-1/a_lam)-1) * b_lam *sqrt(sigma2)

    logR = log_w_density(w_new, mu_new, Non_indicator, sigma2, lam, b_w) +backward_density - log_w_density(w, mu, Non_indicator, sigma2, lam, b_w) - forward_density
    if log(rand(1)[1]) < logR
        w = w_new
        mu = mu_new
        accept = accept + 1
    end
    res = Dict("w" => w, "mu" => mu, "accept_rate" => accept)
    return res
end
function beta_sampler(mu, lam, Non_indicator, theta, sigma2, p)
    m = lam .* sqrt(sigma2)
    beta = quantile.(Exponential.(m),cdf.(Exponential(1/mu),1 ./ m) .* rand(p))
    beta[Non_indicator .== 1] = abs.(theta[Non_indicator .== 1]) .+ mu
    return beta
end
function log_sigma(sigma2,s_beta,p)
    return -((p+3)/2) * log(sigma2)-s_beta/sqrt(sigma2) - p/sigma2 - p*sigma2
end

function sigma2_sampler(sigma2, beta, lam, p, eps_change = 0.01, ub = Inf)
    s_beta = sum(beta./lam)
    accept = 0
    forward_lb = max(sigma2-eps_change,0...)
    forward_ub = min(sigma2+eps_change, ub...)
  
    forward_density = -log(forward_ub-forward_lb)
    sigma2_new = rand(Uniform(forward_lb, forward_ub),1)[1]
  
    backward_lb = max(sigma2_new - eps_change, 0...)
    backward_ub = min(sigma2_new+eps_change,ub...)
    backward_density = -log(backward_ub-backward_lb)

    logR = log_sigma(sigma2_new, s_beta, p) + backward_density - log_sigma(sigma2, s_beta, p) - forward_density
    if log(rand(1)[1]) < logR
        sigma2 = sigma2_new
        accept = 1
    end
    return Dict("sigma2" => sigma2, "accept" => accept)
end
function phicof_sampler(phicof, eps_change, W_A, z, spaind, betaxcov, te2, phitau)
    N = length(phicof)
    accept = 0
    for j in 1:N 
        forward_lb = phicof[j]-eps_change
        forward_ub = phicof[j]+eps_change

        phicof_prop = copy(phicof)
        phicof_prop[j] = rand(Uniform(forward_lb, forward_ub),1)[1]

        logR = log_phicof(j, z, spaind, phicof_prop, betaxcov, te2, phitau, W_A[:,j]) -  log_phicof(j, z, spaind, phicof, betaxcov, te2, phitau, W_A[:,j])
        if log(rand(1)[1]) < logR[1]
            phicof = phicof_prop
            accept = accept + 1
        end
    end
    phicof = phicof .- mean(phicof)
    return Dict("phicof" => phicof, "accept" => accept)
end
    
function log_phicof(j, z, spaind, phicof, betaxcov, te2, phitau, w)
    sum1 = z' * spaind * phicof
    sum2 = te2' * exp.(betaxcov .+ spaind * phicof)
    sum3 = 0.5 * w[j] * phitau * ((w' * phicof)[1]/w[j])^2
    res = sum1 .- sum2 .- sum3
    return res
end
function Credible_Interval(samples, prob)
    p = size(samples)[2]
    nn = size(samples)[1]
    k = round.(nn.*prob,digits=0)
    sample_rank = zeros(nn,p)
    for j in 1:p
        sample_rank[:,j] = rankss(samples[:,j])
    end
    S_left = nn .+ 1 .- minimum(sample_rank,dims=2)
    S_right = maximum(sample_rank,dims=2)
    SS = maximum(hcat(S_left,S_right),dims=2)
    SS = sort(SS,dims = 1)
    jstar = SS[Int.(k)]
    up = zeros(p)
    low = zeros(p)
    CI = Dict()
    ind = 1
    for kk in jstar
        up = copy(up)
        low = copy(low)
        temp = findall(sample_rank .== kk)
        temp2 = findall(sample_rank .== nn + 1 - kk)
        for j in 1:p
            up[j] = samples[temp[j]]
            low[j] = samples[temp2[j]]
        end 
        CI[string(prob[ind])] = Dict("up" => up, "low" => low)
        ind = ind + 1
    end
    return CI
end
function rankss(samples)
    n = length(samples)
    rk = collect(1:n)
    temp = hcat(samples,rk)
    temp_order = hcat(sortslices(temp,dims=1),rk)
    temp = sortslices(temp_order, dims=1, lt=(x,y)->isless(x[2],y[2]))
    return temp[:,3]
end
