#ifndef __BLACK_SCHOLES_CPP
#define __BLACK_SCHOLES_CPP

#define _USE_MATH_DEFINES   // This allows us to use M_PI below
#include <corecrt_math_defines.h>
#include <cmath>  // This is for 'pow'  
#include <iostream>
#include <algorithm>
#include <assert.h>

#include "qmath.h"
#include "black_scholes.h"

using namespace std;

// =================
// ANALYTIC FORMULAE
// =================

// Calculate the European vanilla price based on
// is_call true/false, underlying S, strike K, risk-free rate r, volatility of
// underlying sigma and time to maturity T
double black_scholes_price(const bool is_call, const double S, const double K, const double r, const double v, const double T) {
    return is_call ? 
        (S * norm_cdf(d_j(1, S, K, r, v, T)) - K * exp(-r * T) * norm_cdf(d_j(2, S, K, r, v, T))) :
        (-S * norm_cdf(-d_j(1, S, K, r, v, T)) + K * exp(-r * T) * norm_cdf(-d_j(2, S, K, r, v, T)));
}

// Calculate the European vanilla call Delta
double black_scholes_delta(const bool is_call, const double S, const double K, const double r, const double v, const double T) {
    return is_call ? 
        norm_cdf(d_j(1, S, K, r, v, T)) :
        (norm_cdf(d_j(1, S, K, r, v, T)) - 1);
}

// Calculate the European vanilla call Gamma
double black_scholes_gamma(const bool is_call, const double S, const double K, const double r, const double v, const double T) {
    // is_call does not matter here as identical to call by put-call parity
    return norm_pdf(d_j(1, S, K, r, v, T)) / (S * v * sqrt(T));
}

// Calculate the European vanilla call Vega
double black_scholes_vega(const bool is_call, const double S, const double K, const double r, const double v, const double T) {
    // is_call does not matter here as identical to call by put-call parity
    return S * norm_pdf(d_j(1, S, K, r, v, T)) * sqrt(T);
}

// Calculate the European vanilla call Theta
double black_scholes_theta(const bool is_call, const double S, const double K, const double r, const double v, const double T) {
    return is_call ?
        (-(S * norm_pdf(d_j(1, S, K, r, v, T)) * v) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm_cdf(d_j(2, S, K, r, v, T))) :
        (-(S * norm_pdf(d_j(1, S, K, r, v, T)) * v) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm_cdf(-d_j(2, S, K, r, v, T)));
}

// Calculate the European vanilla call Rho
double black_scholes_rho(const bool is_call, const double S, const double K, const double r, const double v, const double T) {
    return is_call ? 
        (K * T * exp(-r * T) * norm_cdf(d_j(2, S, K, r, v, T))):
        (-T * K * exp(-r * T) * norm_cdf(-d_j(2, S, K, r, v, T)));
}

// =================
// FINITE DIFFERENCE METHOD FORMULAE
// =================

// This uses the forward difference approximation to calculate the Delta of a call option
double black_scholes_delta_fdm(const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_S) {
    return (black_scholes_price(is_call, S + delta_S, K, r, v, T) - black_scholes_price(is_call, S, K, r, v, T)) / delta_S;
}

// This uses the centred difference approximation to calculate the Gamma of a call option
double black_scholes_gamma_fdm(const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_S) {
    return (black_scholes_price(is_call, S + delta_S, K, r, v, T) - 2 * black_scholes_price(is_call, S, K, r, v, T) + black_scholes_price(is_call, S - delta_S, K, r, v, T)) / (delta_S * delta_S);
}

// This uses the forward difference approximation to calculate the Vega of a call option
double black_scholes_vega_fdm(const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_v) {
    return (black_scholes_price(is_call, S, K, r, v + delta_v, T) - black_scholes_price(is_call, S, K, r, v, T)) / delta_v;
}

// This uses the backward difference approximation to calculate the Theta of a call option
double black_scholes_theta_fdm(const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_t) {
    return (black_scholes_price(is_call, S, K, r, v, T - delta_t) - black_scholes_price(is_call, S, K, r, v, T)) / delta_t;
}

// This uses the forward difference approximation to calculate the Rho of a call option
double black_scholes_rho_fdm(const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_r) {
    return (black_scholes_price(is_call, S, K, r + delta_r, v, T) - black_scholes_price(is_call, S, K, r, v, T)) / delta_r;
}

// =================
// MONTE CARLO FORMULAE
// =================

// Simple MC

// Pricing a European vanilla call option with a Monte Carlo method
double monte_carlo_bs_price_simple(const int& num_sims, const bool& is_call, const double& S,
    const double& K, const double& r,
    const double& v, const double& T) {
    double S_adjust = S * exp(T * (r - 0.5 * v * v));
    double S_cur = 0.0;
    double payoff_sum = 0.0;

    for (int i = 0; i < num_sims; i++) {
        double gauss_bm = gaussian_box_muller();
        S_cur = S_adjust * exp(sqrt(v * v * T) * gauss_bm);
        if (is_call) {
            payoff_sum += std::max(S_cur - K, 0.0);
        }
        else {
            payoff_sum += std::max(K - S_cur, 0.0);
        }
    }
    return (payoff_sum / static_cast<double>(num_sims)) * exp(-r * T);
}

// Pricing a European vanilla call option with a Monte Carlo method
// Create three separate paths, each with either an increment, non-
// increment or decrement based on delta_S, the stock path parameter
void monte_carlo_price_simple_3path(const int num_sims, const bool is_call,
    const double S, const double K, const double r, const double v, const double T,
    const double delta_S, const double delta_r, const double delta_v, const double delta_t,
    double& price_Sp, double& price_S, double& price_Sm) {

    // Since we wish to use the same Gaussian random draws for each path, it is
    // necessary to create three separated adjusted stock paths for each 
    // increment/decrement of the asset
    double Sp_adjust = (S + delta_S) * exp((T + delta_t) * ((r + delta_r) - 0.5 * (v + delta_v) * (v + delta_v)));
    double S_adjust = S * exp(T * (r - 0.5 * v * v));
    double Sm_adjust = (S - delta_S) * exp((T - delta_t) * ((r - delta_r) - 0.5 * (v - delta_v) * (v - delta_v)));

    // These will store all three 'current' prices as the Monte Carlo
    // algorithm is carried out
    double Sp_cur = 0.0;
    double S_cur = 0.0;
    double Sm_cur = 0.0;

    // There are three separate pay-off sums for the final price
    double payoff_sum_p = 0.0;
    double payoff_sum = 0.0;
    double payoff_sum_m = 0.0;

    // Loop over the number of simulations
    for (int i = 0; i < num_sims; i++) {
        double gauss_bm = gaussian_box_muller(); // Random gaussian draw

        // Adjust three stock paths 
        double expgauss_p = exp(sqrt((v + delta_v) * (v + delta_v) * (T + delta_t)) * gauss_bm);  // Precalculate
        double expgauss = exp(sqrt(v * v * T) * gauss_bm);  // Precalculate
        double expgauss_m = exp(sqrt((v - delta_v) * (v - delta_v) * (T - delta_t)) * gauss_bm);  // Precalculate
        Sp_cur = Sp_adjust * expgauss_p;
        S_cur = S_adjust * expgauss;
        Sm_cur = Sm_adjust * expgauss_m;

        // Calculate the continual pay-off sum for each increment/decrement
        if (is_call) {
            payoff_sum_p += std::max(Sp_cur - K, 0.0);
            payoff_sum += std::max(S_cur - K, 0.0);
            payoff_sum_m += std::max(Sm_cur - K, 0.0);
        }
        else {
            payoff_sum_p += std::max(K - Sp_cur, 0.0);
            payoff_sum += std::max(K - S_cur, 0.0);
            payoff_sum_m += std::max(K - Sm_cur, 0.0);
        }
    }

    // There are three separate prices
    price_Sp = (payoff_sum_p / static_cast<double>(num_sims)) * exp(-(r + delta_r) * (T + delta_t));
    price_S = (payoff_sum / static_cast<double>(num_sims)) * exp(-r * T);
    price_Sm = (payoff_sum_m / static_cast<double>(num_sims)) * exp(-(r - delta_r) * (T - delta_t));
}

// TODO: to create a MC engine and save /reuse the paths

// These values will be populated via the monte_carlo_call_price function.
// They represent the incremented Sp (S+delta_S), non-incremented S (S) and
// decremented Sm (S-delta_S) prices.
double black_scholes_delta_mc(const int num_sims, const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_S) {
    double price_Sp = 0.0;
    double price_S = 0.0;
    double price_Sm = 0.0;

    // Call the Monte Carlo pricer for each of the three stock paths
    // (We only need two for the Delta)
    monte_carlo_price_simple_3path(num_sims, is_call, S, K, r, v, T, delta_S, 0, 0, 0, price_Sp, price_S, price_Sm);
    return (price_Sp - price_S) / delta_S;
}

double black_scholes_gamma_mc(const int num_sims, const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_S) {
    double price_Sp = 0.0;
    double price_S = 0.0;
    double price_Sm = 0.0;

    // Call the Monte Carlo pricer for each of the three stock paths
    // (We need all three for the Gamma) 
    monte_carlo_price_simple_3path(num_sims, is_call, S, K, r, v, T, delta_S, 0, 0, 0, price_Sp, price_S, price_Sm);
    return (price_Sp - 2 * price_S + price_Sm) / (delta_S * delta_S);
}

double black_scholes_vega_mc(const int num_sims, const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_v) {
    double price_Sp = 0.0;
    double price_S = 0.0;
    double price_Sm = 0.0;

    // Call the Monte Carlo pricer for each of the three stock paths
    // (We only need two for the Vega)
    monte_carlo_price_simple_3path(num_sims, is_call, S, K, r, v, T, 0, 0, delta_v, 0, price_Sp, price_S, price_Sm);
    return (price_Sp - price_S) / delta_v;
}

double black_scholes_theta_mc(const int num_sims, const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_t) {
    double price_Sp = 0.0;
    double price_S = 0.0;
    double price_Sm = 0.0;

    // Call the Monte Carlo pricer for each of the three stock paths
    // (We only need two for the Theta)
    monte_carlo_price_simple_3path(num_sims, is_call, S, K, r, v, T, 0, 0, 0, delta_t, price_Sp, price_S, price_Sm);
    return (price_Sm - price_S) / delta_t;
}

double black_scholes_rho_mc(const int num_sims, const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_r) {
    double price_Sp = 0.0;
    double price_S = 0.0;
    double price_Sm = 0.0;

    // Call the Monte Carlo pricer for each of the three stock paths
    // (We only need two for the Rho)
    monte_carlo_price_simple_3path(num_sims, is_call, S, K, r, v, T, 0, delta_r, 0, 0, price_Sp, price_S, price_Sm);
    return (price_Sp - price_S) / delta_r;
}


#endif