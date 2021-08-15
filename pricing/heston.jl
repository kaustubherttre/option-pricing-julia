"""Heston Model can be described as
dS_t = rS_t dt + √V_t S_t dW_t(1)
dV_t = α(V' - V_t)dt + η√V_t dW_t(2)
dW(1)dW(2) = ρdt
Where, 𝑆𝑡 is the price of the index level in this case at time t, 𝑟 is the risk-free rate r, 𝑉𝑡 is the variance
at time t, 𝑉̅ is the long-term variance V, 𝛼 is the variance mean-reversion speed, 𝜂 is the volatility of
the variance process, 𝑊𝑡(1)
and 𝑊t(2) are two correlated Brownian motion and 𝜌 is the correlation coefficient."""
using SymPy
using PyCall
using Neural
math = pyimport("math")
function f(c,t)
    return q = c^2
    #return s = c-t
end
function heston_C(x,v,τ,r,a,u,b,ρ,σ,ϕ)
    h = (b- ρ*σ*ϕ*1im)
    d = sqrt(((ρ*σ*ϕ*1im)-b)^2-σ^2(2*u*ϕ*1im - ϕ^2))
    g = (h+d)/(h-d)
    D=(h+d)/(σ^2).*(1-exp(d.*τ))./(1-g.*exp(d.*τ))
    k=(1-g*exp(d*τ))/(1-g)
    C=r*ϕ*1im*τ+a/(σ^2).*((h+d)*τ-2*log(k))
    return exp(C+D*v+1im*ϕ*x)
end

function integration_1(ϕ, St, K, r, τ, vt,θ, κ, σ, ρ)
    hes = heston_C(log(St),vt,t,r,κ*θ,0.5,κ-ρ*σ,ρ,σ,ϕ)
    t = exp(1im*ϕ*log(K))*hes/(1im*ϕ)
    return real(t)
end

function integration_2(ϕ, St, K, r, τ, vt,θ, κ, σ, ρ)
    hes = heston_C(log(St),vt,t,r,κ*θ,- 0.5,κ,ρ,σ,ϕ)
    t = exp(-1im*ϕ*log(K))*hes/(1im*ϕ)
    return real(t)
end
@syms ϕ
Π1 = 0.5+ (1/π)*(integrate(integration_1(ϕ, St, K, r, τ, vt,θ, κ, σ, ρ), (ϕ, 0,100)))
Π2 = 0.5+ (1/π)*(integrate(integration_2(ϕ, St, K, r, τ, vt,θ, κ, σ, ρ), (ϕ, 0,100)))
#final price after solving the integration
function heston_price(St, Π1, Π2, r, τ, K)
    return St+Π1-K*exp(-r*τ)*Π2
end
