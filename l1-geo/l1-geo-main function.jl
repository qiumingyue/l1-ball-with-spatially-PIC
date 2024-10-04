
function est_spa(data)
    #data preprocessing
    L = data[:,1]
    R = data[:,2]    
    xcov = data[:,3:(3+p-1)]
    area = data[:,(3+p)]
    status = data[:,((3+p+1))]
    n = length(L)
    ft_max = max(vcat(L,R[isinf.(R).==0])...)
    ft_min = min(vcat(L,R[isinf.(R).==0])...)
    bw_knots = (ft_max - ft_min)/num_knots
    knots = unique(vcat(0,collect(ft_min:bw_knots:ft_max),1.1*ft_max))
    k = length(knots) - 2 + order

    locations = unique(sort(area))
    M = length(locations)
    spaind = zeros(n,M) 
    for i in 1:M
        spaind[area .== locations[i],i] .= 1
    end

    t = copy(R[status .== 0])
    R_true = copy(R[status .!= 0])
    bmst = Mspline(t, order, knots)
    bisL = Ispline(L, order, knots)
    bisR_true = Ispline(R_true, order, knots)
    bisR = zeros(size(bisL))
    bisR[:,status .== 0] = copy(bmst)
    bisR[:,status .!= 0] = copy(bisR_true)
    te3 = copy(bisL)
    te3[:,status .== 1] = bisR[:,status .== 1]
    te3[:,status .== 2] = bisR[:,status .== 2]

    #initial value
    w = 1/p
    theta = ones(p) .* 0.01
    mu = quantile(abs.(theta),1-w)
    lam = ones(p)
    sigma2 = 10
    Non_indicator = Int.(ones(p))
    tau = 1 ./vcat(rand.(InverseGaussian.(lam .* sqrt(sigma2)./abs.(theta),1),1)...)
    gamcoef = rand(Gamma(1,1),k)
    eta = rand(Gamma(a_eta,1/b_eta),1)
    phitau = rand(InverseGamma(a_phitau,b_phitau))
    phicof = ones(M) .* 0.01#rand(MvNormal(phitau * R_phi),1)
    LambdaR = bisR' * gamcoef
    LambdaL = bisL' * gamcoef

    #posterior samplel storage
    #parw = zeros(niter)
    partheta = zeros(niter, p)
    #parbeta = zeros(niter, p)
    #parmu = zeros(niter)
    pargam = zeros(niter, k)
    #pareta = zeros(niter)
    #parsigma = zeros(niter)
    #parlam = zeros(niter,p)
    parNon0 = zeros(niter, p)
    #partau = zeros(niter, p)
    parphitau = zeros(niter)
    parphi = zeros(niter, M)

    #accept rate parameter
    iter = 1
    accept_rate = 0

    eps_w = 0.3
    c_theta = 30000
    c_phi = 20
    #accept_w = zeros(niter)
    #accept_theta = zeros(niter)
    accept_phi = zeros(niter)
    #accept_sigma2 = zeros(niter)
    #accept_non = zeros(niter)

    while iter < niter + 1
        iter % 1000 == 0 ? println(iter) : iter
        #latent variables
        theta_non = theta[Non_indicator.==1]
        xcov_non = xcov[:,Non_indicator.==1]
        z = zeros(n, 1)
        zz = zeros(n, k)
        te1 = exp.(xcov_non * theta_non .+ spaind * phicof) 
        for i = 1:n
            if status[i] == 1.0
                templam1 = LambdaR[i] * te1[i]
                z[i] = max(1,rand(Truncated(Poisson(round(templam1,digits=4)), 1, Inf))...)
                zz[i,:] = rand(Multinomial(Int(z[i]), gamcoef .* bisR[:,i]/sum(gamcoef .* bisR[:,i])),1)
            elseif status[i] == 2.0
                templam1 = (LambdaR[i]-LambdaL[i]) * te1[i]
                z[i] = max(1,rand(Truncated(Poisson(round(templam1,digits=4)), 1, Inf))...)
                zz[i,:] = rand(Multinomial(Int(z[i]), gamcoef .* (bisR[:,i]-bisL[:,i])/sum(gamcoef .* (bisR[:,i]-bisL[:,i]))),1)
            elseif status[i] == 0.0
                z[i] = 1
                zz[i,:] = rand(Multinomial(Int(z[i]), gamcoef .* bisR[:,i]/sum(gamcoef .* bisR[:,i])),1)
            end
        end
        te2 = te3' * gamcoef

        #survival parameter--theta
        lamtau = diagm(1 ./ (lam[Non_indicator .== 1] .^2 .* tau[Non_indicator .== 1]))
        epsilon_hmc_theta = c_theta*iter^(-0.55)/n/length(theta_non)
        theta_ini = HMC_new_sampler(theta_non, epsilon_hmc_theta, accept_rate,log_Posterior_theta, partial_theta, Hessian_theta; xcov = xcov_non, z = z, te2 = te2, spatialcof = spaind * phicof, lamtau = lamtau, sigma2 = sigma2)
        theta[Non_indicator .== 1] = theta_ini["theta"]
        #accept_theta[iter] = theta_ini["accept_rate"]
        #variable selection parameter--tau,Non_indicator,w,mu,beta,lam,sigma2
        tau = tau_sampler(theta, lam, sigma2, Non_indicator, p)
        
        wmu_ini = wmu_sampler(w, Non_indicator, sigma2, lam, a_lam, b_lam, b_w, eps_w, accept_rate)
        w = wmu_ini["w"]
        mu = wmu_ini["mu"]
        #accept_w[iter] = wmu_ini["accept_rate"]

        beta = beta_sampler(mu, lam, Non_indicator, theta, sigma2, p)

        lam = vcat(rand.(InverseGamma.(a_lam + 1, beta ./ sqrt(sigma2) .+ b_lam),1)...)

        sigma2_ini = sigma2_sampler(sigma2, beta, lam, p, 0.5)#min(rand(InverseGamma(p/2,(beta' * diagm(1 ./(2 .* lam .^2 .* tau)) * beta))),1e10)
        sigma2 = sigma2_ini["sigma2"]
        #accept_sigma2[iter] = sigma2_ini["accept"]
        p_non = exp.(-mu./lam./sqrt(sigma2))
        Non_indicator_ini = Nonzeros_sampler(tau, mu, p, lam, theta, sigma2, Non_indicator, 1e-5)#vcat(rand.(Binomial.(1,p_non))...)

        Non_indicator = Non_indicator_ini["Non_indicator"]
        #accept_non[iter] = Non_indicator_ini["accept_rate"]
        
        theta_non = theta[Non_indicator.==1]
        xcov_non = xcov[:,Non_indicator.==1]

        #spatial coefficient--phi
        epsilon_hmc_phi = c_phi * iter^(-0.55)
        phi_true = HMC_new_sampler(phicof,epsilon_hmc_phi,accept_rate,log_Posterior_phi,partial_phi,Hessian_phi;betaxcov = xcov_non * theta_non, z=z, te2=te2, spaind = spaind, R_phi = inv(R_phi), phitau = phitau)
        phicof = phi_true["theta"]
        accept_phi[iter] = phi_true["accept_rate"]
        
        #spatial parameter--phitau
        phitau = rand(InverseGamma(a_phitau + M/2, (phicof' * inv(R_phi) * phicof)[1]/2 + b_phitau),1)[1]


        #spline parameter--gamma,eta
        te1 = exp.(xcov_non * theta_non .+ spaind * phicof) 
        tempa = (1 .+ sum(zz, dims = 1))[1,:]
        tempb = eta .+ (te3 * te1)
        gamcoef = hcat(rand.(Gamma.(tempa,1 ./tempb),1)...)[1,:]

        LambdaR = bisR' * gamcoef
        LambdaL = bisL' * gamcoef

        eta = rand(Gamma(a_eta + k, 1/(b_eta + sum(gamcoef))),1)[1]   
        #=
        if (iter > 1000) + (iter % 500 == 0) == 2
            c_theta = epsilon_change(iter, c_theta, accept_theta)
            eps_w = epsilon_change(iter, eps_w, accept_w)
            c_phi = epsilon_change(iter, c_phi, accept_phi)
            #println(c_beta)
            #println(c_gamma)
            #println(c_cure)
            #println(c_sur)
        end=#

        #parw[iter] = w
        partheta[iter,:] = theta
        #parbeta[iter,:] = beta
        #parmu[iter] = mu
        pargam[iter,:] = gamcoef
        #pareta[iter] = eta
        #parsigma[iter] = sigma2
        #parlam[iter,:] = lam
        parNon0[iter, :] = Non_indicator
        #partau[iter,:] = tau
        parphitau[iter] = phitau
        parphi[iter,:] = phicof
        iter = iter + 1
    end
    result = hcat(partheta[[(5000+2*i) for i = 1:2500],:], parphitau[[(5000+2*i) for i = 1:2500]])#, pargamma_0[[(5000+2*i) for i = 1:2500]],pargamma[[(5000+2*i) for i = 1:2500],:], parOmega_rho[[(5000+2*i) for i = 1:2500]])#, parOmega[[(5000+2*i) for i = 1:2500],:])
    chain_trans = Chains(result, [:theta1, :theta2, :theta3, :theta4, :theta5, :theta6, :theta7, :theta8, :theta9, :theta10,:theta11, :theta12, :theta13, :theta14, :theta15, :theta16, :theta17, :theta18, :theta19, :theta20, :tau])#, :Omega11, :Omega12, :Omega21, :Omega22])
    result_summary = summarystats(chain_trans)
    result_quantile = quantile(chain_trans)
    lower = result_quantile[:,2]
    upper = result_quantile[:,6]
    CI = Credible_Interval(result[:,1:p],prob)
    CI_upper = CI[string(prob)]["up"]
    CI_lower = CI[string(prob)]["low"]
    #lower_HPD = result_quantile[:,2]
    #upper_HPD = result_quantile[:,6]
    estimator = result_summary[:,2]
    mode_ked = zeros(p+1)
    for i in 1:(p+1)
        KDE = KernelDensity.kde(result[:,i])
        tau_max = max(result[:,i]...)
        tau_min = min(result[:,i]...)
        range_tau = collect(tau_min:bw:tau_max)
        tau_den = pdf(KDE,range_tau)
        den_max = max(tau_den...)
        max_ind = tau_den .== den_max
        mode_ked[i] = range_tau' * max_ind
        #HDP = get_HDP(range_tau,tau_den,bw,alpha_CP)
        #lower_HPD[i] = HDP["lower"][1]
        #upper_HPD[i] = HDP["upper"][1]
        #if length(HDP["lower"]) > 1
        #    HPD_more[var_name[i]] = HDP
        #end
    end

    
    res = Dict("mean" => estimator, "lower" => lower, "upper" => upper, "parvar" => result_summary[:,3], "mcse" => result_summary[:,5], "mode" => mode_ked, "non_zeros" => mean(parNon0[[(5000+2*i) for i = 1:2500],:],dims=1)[1,:],"CI_lower" => CI_lower, "CI_upper" => CI_upper)#,#"ssur_acc" => sum(parspatial_sur_accept),"cure_acc" => sum(parspatial_cure_accept),"beta_acc" => sum(parbeta_accept),"gamma_acc" => sum(pargamma_accept),"lower_HPD" => lower_HPD, "upper_HPD" => upper_HPD, "HPD_more" => HPD_more)
    return res
end
