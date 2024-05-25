
#include "pch.h"
#include <Eigen/Dense>
#include "../QBase/matrix.h"
#include "../QBase/qmath.cpp"
#include "../QBase/vanilla_option.cpp"
#include "../QBase/payoff.cpp"

using namespace std;


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

    EXPECT_NEAR(call, 8.5916594188251523, EPISLON);
    EXPECT_NEAR(put, 3.7146018688965654, EPISLON);
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
    EXPECT_NEAR(call, 100.0, EPISLON);
    EXPECT_NEAR(put, 0.0, EPISLON);
}

TEST(TestVanillaOption, TestPutCallParity)
{
    vector<double> strikes = { 90.0, 100.0, 110.0 };
    vector<double> spots = { 90.0, 100.0, 110.0 };
    vector<double> maturities = { 0.0, 1.0 };
    vector<double> sigmas = { 0.0, 0.1 };
    vector<double> rates = { -0.01, 0.0, 0.01 };

    for (double k : strikes) {
        for (double t : maturities) {
            for (double s : spots) {
                for (double sigma : sigmas) {
                    for (double r : rates) {

                        VanillaOption option(k, r, t, s, sigma); // Create the vanilla option
                        double call = option.calc_call_price();
                        double put = option.calc_put_price();
                        double forward = s - k * exp(-r * t);

                        EXPECT_NEAR(call - put, forward, EPISLON);
                    }
                }
            }
        }
    }
}

TEST(TestPayOff, TestCall)
{
    double k = 100.0;
    vector<double> spots = { 80.0, 100.0, 120.0 };
    vector<double> expected_payoff = { 0.0, 0.0, 20.0 };

    EXPECT_EQ(spots.size(), expected_payoff.size());
    for (int i = 0; i < spots.size(); i++)
    {
        double s = spots[i];
        PayOffCall c = PayOffCall(k);
        double payoff = c(s);
        EXPECT_NEAR(payoff, expected_payoff[i], EPISLON);
    }
}

TEST(TestPayOff, TestPut)
{
    double k = 100.0;
    vector<double> spots = { 80.0, 100.0, 120.0 };
    vector<double> expected_payoff = { 20.0, 0.0, 0.0 };

    EXPECT_EQ(spots.size(), expected_payoff.size());
    for (int i = 0; i < spots.size(); i++)
    {
        double s = spots[i];
        PayOffPut p = PayOffPut(k);
        double payoff = p(s);
        EXPECT_NEAR(payoff, expected_payoff[i], EPISLON);
    }
}

TEST(TestPayOff, TestDoubleDigit)
{
    double u = 110.0;
    double d = 90.0;
    vector<double> spots = { 80.0, 100.0, 120.0 };
    vector<double> expected_payoff = { 0.0, 1.0, 0.0 };

    EXPECT_EQ(spots.size(), expected_payoff.size());
    for (int i = 0; i < spots.size(); i++)
    {
        double s = spots[i];
        PayOffDoubleDigital p = PayOffDoubleDigital(u, d);
        double payoff = p(s);
        EXPECT_NEAR(payoff, expected_payoff[i], EPISLON);
    }
}

TEST(TestMatrix, TestSimpleMatrix)
{
    SimpleMatrix<double> sm(4, 4, 1.0);
    // Output values of the SimpleMatrix
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_EQ(sm.value(i, j), 1.0);
        }
    }
}

TEST(TestMatrix, TestQSMatrixAddMatrix)
{
    QSMatrix<double> mat1(10, 10, 1.0);
    QSMatrix<double> mat2(10, 10, 2.0);
    QSMatrix<double> mat3 = mat1 + mat2;

    for (unsigned i = 0; i < mat3.get_rows(); i++) {
        for (unsigned j = 0; j < mat3.get_cols(); j++) {
            EXPECT_EQ(mat3(i, j), 3.0);
        }
    }
}

TEST(TestMatrix, TestQSMatrixMinusMatrix)
{
    QSMatrix<double> mat1(10, 10, 1.0);
    QSMatrix<double> mat2(10, 10, 2.0);
    QSMatrix<double> mat3 = mat1 - mat2;

    for (unsigned i = 0; i < mat3.get_rows(); i++) {
        for (unsigned j = 0; j < mat3.get_cols(); j++) {
            EXPECT_EQ(mat3(i, j), -1.0);
        }
    }
}

TEST(TestMatrix, TestQSMatrixMultMatrix)
{
    QSMatrix<double> mat1(10, 10, 1.0);
    QSMatrix<double> mat2(10, 10, 2.0);
    QSMatrix<double> mat3 = mat1 * mat2;

    for (unsigned i = 0; i < mat3.get_rows(); i++) {
        for (unsigned j = 0; j < mat3.get_cols(); j++) {
            EXPECT_EQ(mat3(i, j), 20.0);
        }
    }
}

TEST(TestMatrix, TestQSMatrixAddScalar)
{
    QSMatrix<double> mat1(10, 10, 1.0);
    double mat2 = 2.0;
    QSMatrix<double> mat3 = mat1 + mat2;

    for (unsigned i = 0; i < mat3.get_rows(); i++) {
        for (unsigned j = 0; j < mat3.get_cols(); j++) {
            EXPECT_EQ(mat3(i, j), 3.0);
        }
    }
}

TEST(TestMatrix, TestQSMatrixMinusScalar)
{
    QSMatrix<double> mat1(10, 10, 1.0);
    double mat2 = 2.0;
    QSMatrix<double> mat3 = mat1 - mat2;

    for (unsigned i = 0; i < mat3.get_rows(); i++) {
        for (unsigned j = 0; j < mat3.get_cols(); j++) {
            EXPECT_EQ(mat3(i, j), -1.0);
        }
    }
}

TEST(TestMatrix, TestQSMatrixMultScalar)
{
    QSMatrix<double> mat1(10, 10, 1.0);
    double mat2 = 2.0;
    QSMatrix<double> mat3 = mat1 * mat2;

    for (unsigned i = 0; i < mat3.get_rows(); i++) {
        for (unsigned j = 0; j < mat3.get_cols(); j++) {
            EXPECT_EQ(mat3(i, j), 2.0);
        }
    }
}

TEST(TestMatrix, TestQSMatrixDivideScalar)
{
    QSMatrix<double> mat1(10, 10, 1.0);
    double mat2 = 2.0;
    QSMatrix<double> mat3 = mat1 / mat2;

    for (unsigned i = 0; i < mat3.get_rows(); i++) {
        for (unsigned j = 0; j < mat3.get_cols(); j++) {
            EXPECT_EQ(mat3(i, j), 0.5);
        }
    }
}

TEST(TestMatrix, TestEigenMatrixReduction)
{
    Eigen::Matrix3d p;
    p << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    // Output the reduction operations
    EXPECT_NEAR(p.sum(), 45.0, EPISLON);
    EXPECT_NEAR(p.prod(), 362880.0, EPISLON);
    EXPECT_NEAR(p.mean(), 5.0, EPISLON);
    EXPECT_NEAR(p.minCoeff(), 1.0, EPISLON);
    EXPECT_NEAR(p.maxCoeff(), 9.0, EPISLON);
    EXPECT_NEAR(p.trace(), 15.0, EPISLON);
}

TEST(TestMatrix, TestEigenMatrixMatrix)
{
    Eigen::Matrix3d p;
    p << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    Eigen::Vector3d r(10, 11, 12);
    Eigen::Vector3d s(13, 14, 15);

    // Matrix/matrix multiplication
    Eigen::Matrix3d p1;
    p1 << 30, 36, 42,
          66, 81, 96,
          102, 126, 150;
    //EXPECT_EQ(p * p, p1);

    // Matrix/vector multiplication
    Eigen::Matrix<double,3,1> p2;
    p2 << 68,
          167,
          266;
    EXPECT_EQ(p * r, p2);

    Eigen::Matrix<double,1,3> p3;
    p3 << 138, 171, 204;
    EXPECT_EQ(r.transpose() * p, p3);

    // Vector/vector multiplication (inner product)
    EXPECT_EQ(r.transpose() * s, 464);
    Eigen::Matrix<double,1,1> p4;
    p4 << 464;
    EXPECT_EQ(r.transpose() * s, p4);
}

TEST(TestMatrix, TestEigenMatrixVector)
{
    // Declare and initialise two 3D vectors
    Eigen::Vector3d r(10, 20, 30);
    Eigen::Vector3d s(40, 50, 60);

    // Apply the 'dot' and 'cross' products 
    EXPECT_EQ(r.dot(s), 3200); // r . s

    Eigen::Matrix<double, 3, 1> rs;
    rs << -300,
          600,
          -300;
    EXPECT_EQ(r.cross(s), rs); // r x s
}

TEST(TestMatrix, TestEigenMatrixTransportation)
{
    // Declare a 3x3 matrix
    Eigen::Matrix3d p;
    p << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    // Output the transpose of p
    Eigen::Matrix3d pt;
    pt << 1, 4, 7,
          2, 5, 8,
          3, 6, 9;

    // In-place transposition
    Eigen::Matrix3d ptt = pt;
    ptt.transposeInPlace();

    EXPECT_EQ(p.transpose(), pt);
    EXPECT_EQ(p, ptt);
}

TEST(TestMatrix, TestEigenMatrixArithmatic)
{  
    // Define two matrices, both 3x3
    Eigen::Matrix3d p;
    Eigen::Matrix3d q;

    // Define two three-dimensional vectors
    // The constructor provides initialisation
    Eigen::Vector3d r(1, 2, 3);
    Eigen::Vector3d s(4, 5, 6);

    // Use the << operator to fill the matrices
    p << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    q << 10, 11, 12,
         13, 14, 15,
         16, 17, 18;

    // Output arithmetic operations for matrices
    Eigen::Matrix3d p_plus_q;
    p_plus_q << 11, 13, 15,
                17, 19, 21,
                23, 25, 27;
    Eigen::Matrix3d p_minus_q;
    p_minus_q << -9, -9, -9,
                 -9, -9, -9,
                 -9, -9, -9;
    EXPECT_EQ(p + q, p_plus_q);
    EXPECT_EQ(p - q, p_minus_q);

    // Output arithmetic operations for vectors
    Eigen::Vector3d r_plus_s(5, 7, 9);
    Eigen::Vector3d r_minus_s(-3, -3, -3);
    EXPECT_EQ(r + s, r_plus_s);
    EXPECT_EQ(r - s, r_minus_s);

    // Multiply and divide by a scalar
    EXPECT_EQ(p * 1.0, p);
    EXPECT_EQ(p / 1.0, p);

    Eigen::Matrix3d zeros = Eigen::Matrix3d::Zero();
    EXPECT_EQ(p * 0.0, zeros);
}