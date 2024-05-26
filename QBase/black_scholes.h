#ifndef __BLACK_SCHOLES_H
#define __BLACK_SCHOLES_H

#define _USE_MATH_DEFINES   // This allows us to use M_PI below
#include <corecrt_math_defines.h>
#include <cmath>  // This is for 'pow'
#include <iostream>
#include <algorithm>
#include <assert.h>

#include "qmath.h"

using namespace std;

// =================
// ANALYTIC FORMULAE
// =================

// Calculate the European vanilla call price based on
// underlying S, strike K, risk-free rate r, volatility of
// underlying sigma and time to maturity T
double black_scholes_price(const bool is_call, const double S, const double K, const double r, const double v, const double T);

// Calculate the European vanilla call Delta
double black_scholes_delta(const bool is_call, const double S, const double K, const double r, const double v, const double T);

// Calculate the European vanilla call Gamma
double black_scholes_gamma(const bool is_call, const double S, const double K, const double r, const double v, const double T);

// Calculate the European vanilla call Vega
double black_scholes_vega(const bool is_call, const double S, const double K, const double r, const double v, const double T);

// Calculate the European vanilla call Theta
double black_scholes_theta(const bool is_call, const double S, const double K, const double r, const double v, const double T);

// Calculate the European vanilla call Rho
double black_scholes_rho(const bool is_call, const double S, const double K, const double r, const double v, const double T);

// =================
// FINITE DIFFERENCE METHOD FORMULAE
// =================

// This uses the forward difference approximation to calculate the Delta of a call option
double black_scholes_delta_fdm(const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_S);

// This uses the centred difference approximation to calculate the Gamma of a call option
double black_scholes_gamma_fdm(const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_S);

// This uses the centred difference approximation to calculate the Vega of a call option
double black_scholes_vega_fdm(const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_v);

// This uses the centred difference approximation to calculate the Theta of a call option
double black_scholes_theta_fdm(const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_t);

// This uses the centred difference approximation to calculate the Rho of a call option
double black_scholes_rho_fdm(const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_r);

// =================
// MONTE CARLO FORMULAE
// =================

// Simple MC

// Pricing a European vanilla call option with a Monte Carlo method
double monte_carlo_bs_price_simple(const int& num_sims, const bool& is_call, const double& S,
    const double& K, const double& r,
    const double& v, const double& T);

// Pricing a European vanilla call option with a Monte Carlo method
// Create three separate paths, each with either an increment, non-
// increment or decrement based on delta_S, the stock path parameter
void monte_carlo_price_simple_3path(const int num_sims, const bool is_call,
    const double S, const double K, const double r, const double v, const double T,
    const double delta_S, const double delta_r, const double delta_v, const double delta_t,
    double& price_Sp, double& price_S, double& price_Sm);

double black_scholes_delta_mc(const int num_sims, const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_S);

double black_scholes_gamma_mc(const int num_sims, const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_S);

double black_scholes_vega_mc(const int num_sims, const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_v);

double black_scholes_theta_mc(const int num_sims, const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_t);

double black_scholes_rho_mc(const int num_sims, const bool is_call, const double S, const double K, const double r, const double v, const double T, const double delta_r);

#endif