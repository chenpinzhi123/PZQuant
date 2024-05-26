
#include "pch.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include "../QBase/matrix.h"
#include "../QBase/qmath.cpp"
#include "../QBase/vanilla_option.cpp"
#include "../QBase/payoff.cpp"

using namespace std;


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

TEST(TestNumericalLinearAlgebra, TestEigenLUDecomposition)
{
    // solve Ax = b; by PA = LU, PAx = Pb and LUx = Pb;
    // --> Ly = Pb; then Ux = y;
    Eigen::MatrixXd A(3,3);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    Eigen::MatrixXd b(3,1);
    b << 0, 1, 2;

    Eigen::PartialPivLU<Eigen::MatrixXd> lu = Eigen::PartialPivLU<Eigen::MatrixXd>(A);
    Eigen::MatrixXd U = lu.matrixLU().triangularView<Eigen::UpLoType::Upper>();
    Eigen::MatrixXd L = lu.matrixLU().triangularView<Eigen::UpLoType::UnitLower>();
    Eigen::MatrixXd P = lu.permutationP(); // .transpose();

    Eigen::MatrixXd F1 = P * b;

    Eigen::MatrixXd y = L.lu().solve(F1);
    Eigen::MatrixXd x = U.lu().solve(y);

    // check that LU = PA
    EXPECT_TRUE((L * U - P * A).norm() < EPISLON);
    // check that Ax = b
    EXPECT_TRUE((A * x - b).norm() < EPISLON);

    // Direct solve; check that A*x1 = b
    Eigen::PartialPivLU<Eigen::MatrixXd> lu1 = Eigen::PartialPivLU<Eigen::MatrixXd>(A);
    Eigen::MatrixXd x1 = lu1.solve(b);
    EXPECT_TRUE((A * x1 - b).norm() < EPISLON);
}

TEST(TestNumericalLinearAlgebra, TestThomasAlgorithm)
{
    // Although thomas_algorithm provides everything necessary to solve
    // a tridiagonal system, it is helpful to wrap it up in a "real world"
    // example. The main function below uses a tridiagonal system from
    // a Boundary Value Problem (BVP). This is the discretisation of the
    // 1D heat equation.

    // Create a Finite Difference Method (FDM) mesh with 13 points
    // using the Crank-Nicolson method to solve the discretised
    // heat equation.
    size_t N = 13;
    double delta_x = 1.0 / static_cast<double>(N);
    double delta_t = 0.001;
    double r = delta_t / (delta_x * delta_x);

    // First we create the vectors to store the coefficients
    std::vector<double> a(N - 1, -r / 2.0);
    std::vector<double> b(N, 1.0 + r);
    std::vector<double> c(N - 1, -r / 2.0);
    std::vector<double> d(N, 0.0);
    std::vector<double> f(N, 0.0);

    // Fill in the current time step initial value
    // vector using three peaks with various amplitudes
    f[5] = 1; f[6] = 2; f[7] = 1;

    // Fill in the current time step vector d
    for (int i = 1; i < N - 1; i++) {
        d[i] = r * 0.5 * f[i + 1] + (1.0 - r) * f[i] + r * 0.5 * f[i - 1];
    }

    // Now we solve the tridiagonal system
    // [a..b..c..][f] = [d]
    thomas_algorithm(a, b, c, d, f);

    // construct matrix to check if equation holds
    Eigen::MatrixXd A(N, N);
    A = Eigen::MatrixXd::Zero(N, N);
    for (int i = 0; i < N; i++) {
        A(i, i) = b[i];
        if (i > 0) {
            A(i, i - 1) = a[i - 1];
        }
        if (i < N - 1) {
            A(i, i + 1) = c[i];
        }
    }
    Eigen::MatrixXd d1(N, 1);
    for (int i = 0; i < N; i++) {
        d1(i, 0) = d[i];
    }
    Eigen::MatrixXd f1(N, 1);
    for (int i = 0; i < N; i++) {
        f1(i, 0) = f[i];
    }

    // Direct solve; check that A*x1 = b
    Eigen::PartialPivLU<Eigen::MatrixXd> lu = Eigen::PartialPivLU<Eigen::MatrixXd>(A);
    Eigen::MatrixXd f2 = lu.solve(d1);

    // check that A * f1 = d1 (f1 is the solution)
    EXPECT_TRUE((A * f1 - d1).norm() < EPISLON);

    // compare with f2 (actual solution from LU decomposition)
    EXPECT_TRUE((A * f2 - d1).norm() < EPISLON);
    EXPECT_TRUE((f1 - f2).norm() < EPISLON);

}

TEST(TestNumericalLinearAlgebra, TestEigenCholeskyDecomposition)
{
    typedef Eigen::Matrix<double, 4, 4> Matrix4x4;

    // Declare a 4x4 matrix with defined entries
    Matrix4x4 p;
    p << 6, 3, 4, 8,
        3, 6, 5, 1,
        4, 5, 10, 7,
        8, 1, 7, 25;

    // Create the L and L^T matrices (LLT)
    Eigen::LLT<Matrix4x4> llt(p);

    // Output L, the lower triangular matrix
    Matrix4x4 l = llt.matrixL();

    // Output L^T, the upper triangular conjugate transpose of L 
    Matrix4x4 lt = l.transpose();

    // Check that LL^T = P
    EXPECT_TRUE((p - l * lt).norm() < EPISLON);
}


TEST(TestNumericalLinearAlgebra, TestEigenQRDecomposition)
{
    // Declare a 3x3 matrix with defined entries
    Eigen::MatrixXd p(3, 3);
    p << 12, -51, 4,
        6, 167, -68,
        -4, 24, -41;

    // Create the Householder QR Matrix object
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(p);
    Eigen::MatrixXd q = qr.householderQ();
    Eigen::MatrixXd r = qr.matrixQR().triangularView<Eigen::Upper>();

    // Check that QR = P
    EXPECT_TRUE((q * r - p).norm() < EPISLON);

    // Check that Q*Q^T = I
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3, 3);
    EXPECT_TRUE((q * q.transpose() - I).norm() < EPISLON);

    // Solve equation for Ax = b <=> QRx = b, Rx = Q^T*b
    Eigen::MatrixXd b(3, 1);
    b << 0,
        1,
        2;
    Eigen::MatrixXd x(3, 1);
    x = qr.solve(b);
    EXPECT_TRUE((p * x - b).norm() < EPISLON);
    EXPECT_TRUE((r * x - q.transpose() * b).norm() < EPISLON);
}

TEST(TestNumericalLinearAlgebra, TestEigenQRDecompositionForLeastSquares)
{
    // Use QR decomposition to get least squares for Xb = y
    Eigen::MatrixXd X(6, 2);
    X << 10, 2,
        5, -2,
        7, 4,
        8, 0,
        12, -3,
        11, -6;
    Eigen::MatrixXd y(6, 1);
    y << 6, 3, 0, 5, 13, 16;

    // compare b(numerical values), b1(from LU), and b2(from QR)
    Eigen::MatrixXd b(2, 1);
    b << 0.737872006794942913998625,
        -1.1851621965959950877334;
    Eigen::MatrixXd b1(2, 1);
    Eigen::MatrixXd b2(2, 1);

    // Slow: LU decomposition, use X^T*X*b = X^t*y to solve it first
    Eigen::PartialPivLU<Eigen::MatrixXd> lu1 = Eigen::PartialPivLU<Eigen::MatrixXd>(X.transpose() * X);
    b1 = lu1.solve(X.transpose() * y);

    EXPECT_TRUE((b - b1).norm() < EPISLON);

    // Faster: QR decomposition, X = QR;
    // X^T*X*b = X^T*y <=> R^T*Q^T*Q*R*b = R^T*Q^T*y
    // <=> R^T*R*b = R^T*Q^T*y <=> R^T*(R*b - Q^T*y) = 0, R is non-zero upper triangular matrix
    // <=> R*b - Q^T*y = 0
    // - solve as R*b = Q^T*y, with R / Q are the first k vector / square matrix
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(X);
    Eigen::MatrixXd q = qr.householderQ();
    Eigen::MatrixXd r = qr.matrixQR().triangularView<Eigen::Upper>();    
    Eigen::MatrixXd qty = q.transpose() * y;

    Eigen::MatrixXd r_ = r.block(0, 0, 2, 2);
    Eigen::MatrixXd qty_ = qty.block(0, 0, 2, 1);

    b2 = r_.triangularView<Eigen::Upper>().solve(qty_);
    EXPECT_TRUE((b - b2).norm() < EPISLON);
}


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


TEST(TestVanillaOption, TestSimpleMonteCarlo)
{
    double k = 100.0;
    double maturity = 1.0;
    double s = 100.0;
    double sigma = 0.15;
    double r = 0.05;

    VanillaOption option(k, r, maturity, s, sigma); // Create the vanilla option

    // Calculate the call and put prices (analytical)
    double call = option.calc_call_price();
    double put = option.calc_put_price();

    srand(1);

    const int N1 = 1000;
    double call_mc1 = option.calc_call_price_monte_carlo(N1);
    double put_mc1 = option.calc_put_price_monte_carlo(N1);

    const int N2 = 100000;
    double call_mc2 = option.calc_call_price_monte_carlo(N2);
    double put_mc2 = option.calc_put_price_monte_carlo(N2);

    // Analytical
    EXPECT_NEAR(call, 8.5916594188251523, EPISLON);
    EXPECT_NEAR(put, 3.7146018688965654, EPISLON);

    // Simple MC with N = 1,000
    EXPECT_NEAR(call_mc1, 8.6790390302383322, EPISLON);
    EXPECT_NEAR(put_mc1, 3.60328962393, EPISLON);

    // Simple MC with N = 100,000
    EXPECT_NEAR(call_mc2, 8.6567854960174575, EPISLON);
    EXPECT_NEAR(put_mc2, 3.7356994889424135, EPISLON);

}