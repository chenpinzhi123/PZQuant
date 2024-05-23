// QBase.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "vanilla_option.h"

int main()
{
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
