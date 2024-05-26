#ifndef __QMATH_CPP
#define __QMATH_CPP

#include "qmath.h"

// probabilistic utilities

// Standard normal probability density function
double norm_pdf(const double x)
{
    return (1.0 / (pow(2 * M_PI, 0.5))) * exp(-0.5 * x * x);
}

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

// A simple implementation of the Box-Muller algorithm, used to generate
// gaussian random numbers - necessary for the Monte Carlo method below
// Note that C++11 actually provides std::normal_distribution<> in 
// the <random> library, which can be used instead of this function
double gaussian_box_muller() {
    double x = 0.0;
    double y = 0.0;
    double euclid_sq = 0.0;

    // Continue generating two uniform random variables
    // until the square of their "euclidean distance" 
    // is less than unity
    do {
        x = 2.0 * rand() / static_cast<double>(RAND_MAX) - 1;
        y = 2.0 * rand() / static_cast<double>(RAND_MAX) - 1;
        euclid_sq = x * x + y * y;
    } while (euclid_sq >= 1.0);

    return x * sqrt(-2 * log(euclid_sq) / euclid_sq);
}

// This calculates d_j, for j in {1,2}. This term appears in the closed
// form solution for the European call or put price
double d_j(const int j, const double S, const double K, const double r,
    const double v, const double T)
{
    return (log(S / K) + (r + (pow(-1, j - 1)) * 0.5 * v * v) * T) / (v * (pow(T, 0.5)));
}

// linear algebra utilities

// Vectors a, b, c and d are const. They will not be modified
// by the function. Vector f (the solution vector) is non-const
// and thus will be calculated and updated by the function.
void thomas_algorithm(const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d,
    std::vector<double>& f) {
    size_t N = d.size();

    // Create the temporary vectors
    // Note that this is inefficient as it is possible to call
    // this function many times. A better implementation would
    // pass these temporary matrices by non-const reference to
    // save excess allocation and deallocation
    std::vector<double> c_star(N, 0.0);
    std::vector<double> d_star(N, 0.0);

    // This updates the coefficients in the first row
    // Note that we should be checking for division by zero here
    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    // Create the c_star and d_star coefficients in the forward sweep
    for (int i = 1; i < N; i++) { // ??????????????????????????/
        double m = 1.0 / (b[i] - a[i-1] * c_star[i - 1]); // a[i] -> a[i-1]
        if (i < N - 1) {
            c_star[i] = c[i] * m;
        }
        d_star[i] = (d[i] - a[i-1] * d_star[i - 1]) * m;  // a[i] -> a[i-1]
    }

    // This is the reverse sweep, used to update the solution vector f
    f[N - 1] = d_star[N - 1]; // add
    for (int i = N - 1; i-- > 0; ) {
        f[i] = d_star[i] - c_star[i] * f[i + 1]; // d[i+1] -> f[i+1]
    }
}
#endif
