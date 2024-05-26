#ifndef __PAY_OFF_CPP
#define __PAY_OFF_CPP

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

#endif
