#ifndef __VANILLA_OPTION_CPP
#define __VANILLA_OPTION_CPP

#define _USE_MATH_DEFINES   // This allows us to use M_PI below
#include <corecrt_math_defines.h>
#include <cmath>  // This is for 'pow'  
#include <iostream>
#include <algorithm>
#include <assert.h>

#include "vanilla_option.h"

using namespace std;

double norm_cdf(const double x)
{
    // An approximation to the cumulative distribution function
    // for the standard normal distribution
    // Note: This is a recursive function
    if (x >= 0.0)
    {
        double k = 1.0 / (1.0 + 0.2316419 * x); 
        double k_sum = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + 1.330274429 * k))));
        return (1.0 - (1.0 / (pow(2 * M_PI, 0.5))) * exp(-0.5 * x * x) * k_sum);
    }
    else
    {
        return 1.0 - norm_cdf(-x);
    }
}

// Parameter constructor
VanillaOption::VanillaOption(const double _K, const double _r, 
                             const double _T, const double _S, 
                             const double _sigma) :
    K(_K), r(_r), T(_T), S(_S), sigma(_sigma)
{
}

// Destructor
VanillaOption::~VanillaOption() {
    // Empty, as the compiler does the work of cleaning up the simple types for us
}

// Copies the member data 
void VanillaOption::copy(const VanillaOption& rhs)
{
    K = rhs.getK();
    r = rhs.getr();
    T = rhs.getT();
    S = rhs.getS();
    sigma = rhs.getsigma();
}

// Copy constructor
VanillaOption::VanillaOption(const VanillaOption& rhs)
{
    copy(rhs);
}

// Assignment operator
VanillaOption& VanillaOption::operator=(const VanillaOption& rhs)
{
    if (this == &rhs)
        return *this;
    copy(rhs);
    return *this;
}


// Public access for the strike price, K
double VanillaOption::getK() const { return K; }

// Public access for the risk-free rate, r
double VanillaOption::getr() const { return r; }

// Public access for the time to maturity, T
double VanillaOption::getT() const { return T; }

// Public access for the spot price, S
double VanillaOption::getS() const { return S; }

// Public access for the volatility of the asset, sigma
double VanillaOption::getsigma() const { return sigma; }

// Calculate the vanilla call price (uses 'N', an approximation to
// the cumulative distribution function of the normal distribution)
double VanillaOption::calc_call_price() const
{
    assert(T >= 0 && sigma >= 0); // time to expiry and sigma must be >= 0.0
    if (T < EPISLON || sigma < EPISLON)
    {
        return max(S - K * exp(-r * T), 0.0);
    }
    double sigma_sqrt_T = sigma * sqrt(T);
    double d_1 = ( log(S/K) + (r + sigma * sigma * 0.5 ) * T ) / sigma_sqrt_T;
    double d_2 = d_1 - sigma_sqrt_T;
    return S * norm_cdf(d_1) - K * exp(-r*T) * norm_cdf(d_2);
}

// Calculate the vanilla put price (uses 'N', an approximation to
// the cumulative distribution function of the normal distribution)  
double VanillaOption::calc_put_price() const
{
    assert(T >= 0 && sigma >= 0); // time to expiry and sigma must be >= 0.0
    if (T < EPISLON || sigma < EPISLON)
    {
        return max(K * exp(-r * T) - S, 0.0);
    }
    double sigma_sqrt_T = sigma * sqrt(T);
    double d_1 = ( log(S/K) + (r + sigma * sigma * 0.5 ) * T ) / sigma_sqrt_T;
    double d_2 = d_1 - sigma_sqrt_T;
    return K * exp(-r*T) * norm_cdf(-d_2) - S * norm_cdf(-d_1);
}

#endif