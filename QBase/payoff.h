#ifndef __PAY_OFF_H
#define __PAY_OFF_H

#include <algorithm> // This is needed for the std::max comparison function, used in the pay-off calculations
#include <vector>

class PayOff
{
public:
	PayOff() {};
	virtual ~PayOff() {};
	virtual double operator() (const double S) const = 0;
};

class PayOffCall : public PayOff
{
private:
	double K;
public:
	PayOffCall(const double K_);
	virtual ~PayOffCall() {};
	virtual double operator() (const double S) const;
};

class PayOffPut : public PayOff
{
private:
	double K;
public:
	PayOffPut(const double K_);
	virtual ~PayOffPut() {};
	virtual double operator() (const double S) const;
};

class PayOffDoubleDigital : public PayOff {
private:
	double U;  // Upper strike price
	double D;  // Lower strike price

public:
	// Two strike parameters for constructor
	PayOffDoubleDigital(const double _U, const double _D);

	// Destructor
	virtual ~PayOffDoubleDigital();

	// Pay-off is 1 if spot within strike barriers, 0 otherwise
	virtual double operator() (const double S) const;
};

class PayOffAsianOption {
protected:
	PayOff* pay_off;  // Pay-off class (in this instance call or put) 

public:
	PayOffAsianOption(PayOff* _pay_off);
	virtual ~PayOffAsianOption() {};

	// Pure virtual pay-off operator (this will determine arithmetic or geometric)
	virtual double pay_off_price(const std::vector<double>& spot_prices) const = 0;
};

class PayOffAsianOptionArithmetic : public PayOffAsianOption {
public:
	PayOffAsianOptionArithmetic(PayOff* _pay_off);
	virtual ~PayOffAsianOptionArithmetic() {};

	// Overide the pure virtual function to produce arithmetic Asian Options
	virtual double pay_off_price(const std::vector<double>& spot_prices) const;
};

class PayOffAsianOptionGeometric : public PayOffAsianOption {
public:
	PayOffAsianOptionGeometric(PayOff* _pay_off);
	virtual ~PayOffAsianOptionGeometric() {};

	// Overide the pure virtual function to produce geometric Asian Options
	virtual double pay_off_price(const std::vector<double>& spot_prices) const;
};



#endif