
#include "pch.h"
#include "../QBase/vanilla_option.cpp"

using namespace std;
const double TEST_EPISLON = 1e-12;


TEST(TestVanillaOption, TestBS1)
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

    EXPECT_NEAR(call, 8.5916594188251523, TEST_EPISLON);
    EXPECT_NEAR(put, 3.7146018688965654, TEST_EPISLON);
}

TEST(TestVanillaOption, TestBS2)
{
    double k = 0.0;
    double maturity = 1.0;
    double s = 100.0;
    double sigma = 0.15;
    double r = 0.00;

    VanillaOption option(k, r, maturity, s, sigma); // Create the vanilla option

    // Calculate the call and put prices
    double call = option.calc_call_price();
    double put = option.calc_put_price();

    // strike = 0, call = 100 and put = 0
    EXPECT_NEAR(call, 100.0, TEST_EPISLON);
    EXPECT_NEAR(put, 0.0, TEST_EPISLON);
}

TEST(TestVanillaOption, TestPutCallParity)
{
    vector<double> strikes = { 50.0, 90.0, 95.0, 100.0, 105.0, 110.0, 150.0 };
    vector<double> spots = { 50.0, 90.0, 95.0, 100.0, 105.0, 110.0, 150.0 };
    vector<double> maturities = { 1.0 / 252, 1.0 / 12, 0.5, 1.0, 2.0, 3.0, 5.0 };
    vector<double> sigmas = { 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5 };
    vector<double> rates = { -0.01, 0.0, 0.01, 0.05, 0.10 };

    for (double k : strikes) {
        for (double t : maturities) {
            for (double s : spots) {
                for (double sigma : sigmas) {
                    for (double r : rates) {

                        VanillaOption option(k, r, t, s, sigma); // Create the vanilla option
                        double call = option.calc_call_price();
                        double put = option.calc_put_price();
                        double forward = s - k * exp(-r * t);

                        EXPECT_NEAR(call - put, forward, TEST_EPISLON);
                    }
                }
            }
        }
    }
}