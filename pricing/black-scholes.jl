# The Black Scholes model is one of the most important financial tools to solve for price of a option. The BS model works by making the following assumptions:
#     1. The volatility and the risk-free rate are constant
#     2. The stock price is a stochastic process follows GBM
#     3. Limited to European Options
#
# Black Scholes model -
#
# S -> Price of underlying stock
# K -> Strike Price
# τ -> Time to Expiry in years
# r -> risk free rate
# σ -> Volatility

#Brent's Method for optimization

function brent(f::Function, x0::Number, x1::Number, args::Tuple=();
               xtol::AbstractFloat=1e-7, ytol=2eps(Float64),
               maxiter::Integer=50)
    EPS = eps(Float64)
    y0 = f(x0,args...)
    y1 = f(x1,args...)
    if abs(y0) < abs(y1)
        # Swap lower and upper bounds.
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    end
    x2 = x0
    y2 = y0
    x3 = x2
    bisection = true
    for _ in 1:maxiter
        # x-tolerance.
        if abs(x1-x0) < xtol
            return x1
        end

        # Use inverse quadratic interpolation if f(x0)!=f(x1)!=f(x2)
        # and linear interpolation (secant method) otherwise.
        if abs(y0-y2) > ytol && abs(y1-y2) > ytol
            x = x0*y1*y2/((y0-y1)*(y0-y2)) +
                x1*y0*y2/((y1-y0)*(y1-y2)) +
                x2*y0*y1/((y2-y0)*(y2-y1))
        else
            x = x1 - y1 * (x1-x0)/(y1-y0)
        end

        # Use bisection method if satisfies the conditions.
        delta = abs(2EPS*abs(x1))
        min1 = abs(x-x1)
        min2 = abs(x1-x2)
        min3 = abs(x2-x3)
        if (x < (3x0+x1)/4 && x > x1) ||
           (bisection && min1 >= min2/2) ||
           (!bisection && min1 >= min3/2) ||
           (bisection && min2 < delta) ||
           (!bisection && min3 < delta)
            x = (x0+x1)/2
            bisection = true
        else
            bisection = false
        end

        y = f(x,args...)
        # y-tolerance.
        if abs(y) < ytol
            return x
        end
        x3 = x2
        x2 = x1
        if sign(y0) != sign(y)
            x1 = x
            y1 = y
        else
            x0 = x
            y0 = y
        end
        if abs(y0) < abs(y1)
            # Swap lower and upper bounds.
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        end
    end
    error("Max iteration exceeded")
end
# PayOff function
using Distributions
function call_op_payoff(S,K)
    return max(S-K,0)
end
function norm_cdf(x)
    return cdf(Normal(0, 1), x)
end

function bs_call_val(S,K,r,τ,σ)
    d1 = (1.0/(σ * sqrt(τ))) * (log(S/K) + (r+ 0.5* σ^2.0) * τ)
    d2 = d1 - (σ * sqrt(τ))
    return norm_cdf(d1) * S - norm_cdf(d2) * K * exp(-r * τ)
end

function bs_put_val(S,K,r,τ,σ)
    d1 = (1.0/(σ * sqrt(τ))) * (log(S/K) + (r+ 0.5* σ^2.0) * τ)
    d2 = d1 - (σ * sqrt(τ))
    return norm_cdf(-d2) * K  * exp(-r * τ) - (norm_cdf(-d1)*S)
end
function call_err(S,K,r,τ,σ,call_price)
    return call_price - bs_call_val(S,K,r,τ,σ)
end
function put_err(S,K,r,τ,σ,put_price)
    return put_price - bs_put_val(S,K,r,τ,σ)
end
#Bug in brents method
function call_imp_vol(S,K,r,τ,call_price, a = -2, b = 2, tol = 1e-6)
    dum_S = S
    dum_K = K
    dum_τ = τ
    dum_r = r
    dum_C = call_price

    function func()
        return call_err(dum_S,dum_K,dum_r,dum_τ,σ,dum_C)
    end
    try
        res = brent(func, x0 = a , x1 = b)
        return res
    catch
        return NaN
end
function put_imp_vol(S,K,r,τ,put_price, a = -2, b = 2, tol = 1e-6)
    dum_S = S
    dum_K = K
    dum_τ = τ
    dum_r = r
    dum_P = put_price

    function func()
        return put_err(dum_S,dum_K,dum_r,dum_τ,σ,dum_C)
    end
    try
        res = brent(func, x0 = a , x1 = b)
        return res
    catch
        return NaN
end
# call_err(100,90,0.001,0.6,0.01,11)
# call_imp_vol(100,90,0.001,0.6,1100)
#

#Greeks
function f(t)
     return exp( -0.5 * t * t)/ (sqrt(2.0 * π))
 end
function δ(S,K,r,τ,σ)
    d = (1.0/(σ * sqrt(τ))) * (log(S/K) + (r + 0.5+ σ^2) * τ)
    return norm_cdf(d)
end
function γ(S,K,r,τ,σ)
    d = (1.0/(σ * sqrt(τ))) * (log(S/K) + (r + 0.5+ σ^2) * τ)
    return f(d) / (S * σ * sqrt(τ))
end
function vega(S,K,r,τ,σ)
    d = (1.0/(σ * sqrt(τ))) * (log(S/K) + (r + 0.5+ σ^2) * τ)
    return  (S * func(d) * sqrt(τ)) / 100.0
end
function θ(S,K,r,τ,σ)
    d1  = (1.0/(σ * sqrt(τ))) * (log(S/K) + (r + 0.5+ σ^2) * τ)
    d2 = d1 - (σ * np.sqrt(τ))
    theta = -((S * f(d1) * σ) / (2.0 * sqrt(t))) + (r * K * exp(-r * τ) * norm_cdf(-d2))
    return theta/365.0
end
function ρ(S,K,r,τ,σ)
    d1  = (1.0/(σ * sqrt(τ))) * (log(S/K) + (r + 0.5+ σ^2) * τ)
    d2 = d1 - (σ * np.sqrt(τ))
    rho = K * τ * exp(-r * τ) * norm_cdf(d2)
    return rho/100.0
end
function moneyness(S,K)
    return S/K
end
