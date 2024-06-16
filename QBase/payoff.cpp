#ifndef __PAY_OFF_CPP
#define __PAY_OFF_CPP

#include <numeric>  // Necessary for std::accumulate
#include <cmath>  // For log/exp functions
#include "payoff.h"

// ==========
// PayOffCall
// ==========

// Constructor with single strike parameter
PayOffCall::PayOffCall(const double _K) {
	K = _K;
}

// Over-ridden operator() method, which turns PayOffCall into a function object
double PayOffCall::operator() (const double S) const {
	return std::max(S - K, 0.0);
}

// =========
// PayOffPut
// =========

// Constructor with single strike parameter
PayOffPut::PayOffPut(const double _K) {
	K = _K;
}

// Over-ridden operator() method, which turns PayOffPut into a function object
double PayOffPut::operator() (const double S) const {
	return std::max(K - S, 0.0); // Standard European put pay-off
}

// Constructor with two strike parameters, upper and lower barrier
PayOffDoubleDigital::PayOffDoubleDigital(const double _U, const double _D) {
	U = _U;
	D = _D;
}

// Destructor
PayOffDoubleDigital::~PayOffDoubleDigital() {}

// Over-ridden operator() method, which turns 
// PayOffDoubleDigital into a function object
double PayOffDoubleDigital::operator() (const double S) const {
	if (S >= D && S <= U) {
		return 1.0;
	}
	else {
		return 0.0;
	}
}

// =====================
// AsianOptionArithmetic
// =====================

PayOffAsianOption::PayOffAsianOption(PayOff* _pay_off) : pay_off(_pay_off) {}

// =====================  
// AsianOptionArithmetic
// =====================  

PayOffAsianOptionArithmetic::PayOffAsianOptionArithmetic(PayOff* _pay_off) : PayOffAsianOption(_pay_off) {}

// Arithmetic mean pay-off price
double PayOffAsianOptionArithmetic::pay_off_price(const std::vector<double>& spot_prices) const {
	unsigned num_times = spot_prices.size();
	double sum = std::accumulate(spot_prices.begin(), spot_prices.end(), 0);
	double arith_mean = sum / static_cast<double>(num_times);
	return (*pay_off)(arith_mean);
}

// ====================
// AsianOptionGeometric
// ====================

PayOffAsianOptionGeometric::PayOffAsianOptionGeometric(PayOff* _pay_off) : PayOffAsianOption(_pay_off) {}

// Geometric mean pay-off price
double PayOffAsianOptionGeometric::pay_off_price(const std::vector<double>& spot_prices) const {
	unsigned num_times = spot_prices.size();
	double log_sum = 0.0;
	for (int i = 0; i < spot_prices.size(); i++) {
		log_sum += log(spot_prices[i]);
	}
	double geom_mean = exp(log_sum / static_cast<double>(num_times));
	return (*pay_off)(geom_mean);
}

#endif
