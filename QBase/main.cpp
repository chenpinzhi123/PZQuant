// QBase.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "matrix.h"
#include "qmath.h"
#include "vanilla_option.h"
#include "payoff.h"

int main()
{
    /*
    double k = 100.0;
    double maturity = 1.0;
    double s = 100.0;
    double sigma = 0.15;
    double r = 0.05;

    VanillaOption option(k, r, maturity, s, sigma); // Create the vanilla option

    // Calculate the call and put prices
    double call = option.calc_call_price();
    double put = option.calc_put_price();

    // Output the option parameters
    std::cout << "Strike, K: " << option.getK() << std::endl;
    std::cout << "Risk-free rate, r: " << option.getr() << std::endl;
    std::cout << "Time to maturity, T: " << option.getT() << std::endl;
    std::cout << "Spot price, S: " << option.getS() << std::endl;
    std::cout << "Volatility of asset, sigma: " << option.getsigma() << std::endl;

    // Output the option prices
    std::cout << "Call Price: " << call << std::endl;
    std::cout << "Put Price: " << put << std::endl;

    double D = 10.0;  // Lower strike
    double U = 20.0;  // Upper strike

    PayOffDoubleDigital pay(U, D);  // Create the double digital payoff

    // Output the payoff for various spot prices
    std::cout << "Spot = 5.0 : " << pay(5.0) << std::endl;
    std::cout << "Spot = 15.0: " << pay(15.0) << std::endl;
    std::cout << "Spot = 25.0: " << pay(25.0) << std::endl;

    SimpleMatrix<double> sm(4, 4, 0.0);

    // Output values of the SimpleMatrix
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << sm.value(i, j) << "\t";
        }
        std::cout << std::endl;
    }
    */

    QSMatrix<double> mat1(10, 10, 1.0);
    QSMatrix<double> mat2(10, 10, 2.0);
    QSMatrix<double> mat3 = mat1 + mat2;

    for (int i = 0; i < mat3.get_rows(); i++) {
        for (int j = 0; j < mat3.get_cols(); j++) {
            std::cout << mat3(i, j) << ", ";
        }
        std::cout << std::endl;
    }


    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
