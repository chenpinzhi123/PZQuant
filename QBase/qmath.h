#ifndef __QMATH_H
#define __QMATH_H

#define _USE_MATH_DEFINES   // This allows us to use M_PI below
#include <corecrt_math_defines.h>
#include <cmath>  // This is for 'pow'
#include <vector>
using namespace std;

const double EPISLON = 1e-12;

// probabilistic utilities

double norm_pdf(const double x);
double norm_cdf(const double x);
double gaussian_box_muller();

double d_j(const int j, const double S, const double K, const double r,
    const double v, const double T);

// linear algebra utilities

// Vectors a, b, c and d are const. They will not be modified
// by the function. Vector f (the solution vector) is non-const
// and thus will be calculated and updated by the function.
void thomas_algorithm(const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d,
    std::vector<double>& f);

#endif