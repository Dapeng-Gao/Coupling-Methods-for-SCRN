#define NDEBUG
#define EIGEN_NO_DEBUG
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#define _USE_MATH_DEFINES

#include "Reflection_coupling.h"
#include "Prob.h"
#include "CRN.h"

#include <omp.h>
#include <iostream>
#include <random>
#include <math.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>


ReflectionCoupling::ReflectionCoupling()
{
    BM1 = MatrixXd::Zero(N_res, CRN::k);
    BM2 = MatrixXd::Zero(N_res, CRN::k);
    TT1 = MatrixXd::Zero(N_res, CRN::k);
    TT2 = MatrixXd::Zero(N_res, CRN::k);
}

// Reflection coupling of Diffusion approximation
double ReflectionCoupling::simulate(const VectorXd& X1_start, const VectorXd& X2_start, mt19937 &mt, normal_distribution<double> &nm, 
                        uniform_real_distribution<double> &u)
{
    // ofstream traj1("traj_X1.txt", ios::app), traj2("traj_X2.txt", ios::app);
    // traj1 << X1_start.transpose() << endl;
    // traj2 << X2_start.transpose() << endl;

    // random_device rd;
    // mt19937 mt(rd());
    // normal_distribution<double> nm(0.0, 1.0);
    // uniform_real_distribution<double> u(0, 1.0);

    VectorXd X1 = X1_start;
    VectorXd X2 = X2_start;
    int count = 0, flag = 0;
    // int count_out = 0;
    if(allPositive(X1_start) && allPositive(X2_start))
    {

        while (count < 1e8 && flag == 0)
        {
            count++;

            MatrixXd A = Cov((X1 + X2) / 2);
            MatrixXd A1 = Cov(X1);
            MatrixXd A2 = Cov(X2);
            MatrixXd B = new_sigma(A);
            MatrixXd B1 = new_sigma(A1);
            MatrixXd B2 = new_sigma(A2);

            MatrixXd M1 = M(X1);
            MatrixXd M2 = M(X2);
            VectorXd R1 = sqrtrate(X1);
            VectorXd R2 = sqrtrate(X2);

            VectorXd rnd1(CRN::d), rnd2(CRN::d);
            VectorXd W1(CRN::k), W2(CRN::k);

            // Reflection coupling

            for (int i = 0; i < CRN::d; ++i) {
                rnd1(i) = nm(mt);
            }
            rnd2 = reflection(X1, X2, rnd1, B);
            // cout << "rnd1 = " << rnd1.transpose() << endl;
            // cout << "rnd2 = " << rnd2.transpose() << endl;
            // CRN SIR_R;
            // Update diffusion Status
            // cout << "drift1 = " << drift(X1).transpose() << endl;
            // cout << "drift2 = " << drift(X2).transpose() << endl;
            X1 += CRN::h * drift(X1);
            X2 += CRN::h * drift(X2);

            // cout << "X1 = " << X1.transpose() << endl;
            // cout << "X2 = " << X2.transpose() << endl;
            // system("pause");
                     
            // B*increment
            VectorXd tmp1 = B1 * rnd1 * sqrt(CRN::h);
            VectorXd tmp2 = B2 * rnd2 * sqrt(CRN::h);

        // maximal coupling of diffusion
            if (allPositive(X1) && allPositive(X2))
            {
                // cout << "test maximal coupling" << endl;
                VectorXd diff = X1 - X2;
                // double threshold = 3 * sqrt(CRN::h);
                double threshold = 3 * sqrt(CRN::h) * max(B1.norm(), B2.norm());
                // cout << B1 << endl;
                // cout << B2 << endl;
                // cout << B1.norm() << endl;
                // cout << B2.norm() << endl;

                if (diff.norm() < threshold)
                {
                    // cout << "Entering maximal coupling" << endl;
                    for (int i = 0; i < CRN::d; ++i) 
                    {
                        rnd2(i) = nm(mt);
                    }
                    tmp2 = B2 * rnd2 * sqrt(CRN::h);

                    double px = Prob::normpdf(X1 + tmp1, X1, A1);
                    double qx = Prob::normpdf(X1 + tmp1, X2, A2);
                    double py = Prob::normpdf(X2 + tmp2, X1, A1);
                    double qy = Prob::normpdf(X2 + tmp2, X2, A2);

                // Successful coupling
                    if (u(mt) * px < qx)
                    {
                        flag = 1;
                        X1 += tmp1;
                        X2 = X1; // coupling point
                        BM2.row(count) = BM1.row(count);
                    }

                // failure
                    else
                    {
                        X1 += tmp1;
                        int trial = 0;
                        
                        while (u(mt) * qy <= py && trial < 1000)
                        {
                            for (int i = 0; i < CRN::d; ++i) 
                            {
                                rnd2(i) = nm(mt);
                            }
                            tmp2 = B2 * rnd2 * sqrt(CRN::h);
                            qy = Prob::normpdf(X2 + tmp2, X2, A2);
                            py = Prob::normpdf(X2 + tmp2, X1, A1);
                            trial ++;
                        }
                        X2 += tmp2;
                        
                    }
                }
                else
                {
                    X1 += tmp1;
                    X2 += tmp2;
                    // cout << "fail coupling" << endl;
                    // cout << "X1 = " << X1.transpose() << endl;
                    // cout << "X2 = " << X2.transpose() << endl;
                    // system("pause");
                }
                // cout <<"begin solveW"<< endl;
            // go back to four normal variable
                VectorXd rnd_bar(CRN::k);
                for (int i = 0; i < CRN::k; ++i) {
                    rnd_bar(i) = nm(mt); 
                }

                W1 = solveW(M1, tmp1, rnd_bar * sqrt(CRN::h));
                W2 = solveW(M2, tmp2, rnd_bar * sqrt(CRN::h));

                // W11 << W1.transpose() << endl;
                // W22 << W2.transpose() << endl;

            // Diffusion with k W
                BM1.row(count) = BM1.row(count - 1) + W1.cwiseProduct(R1).transpose();
                TT1.row(count) = TT1.row(count - 1) + R1.cwiseProduct(R1).transpose() * CRN::h;
                BM2.row(count) = BM2.row(count - 1) + W2.cwiseProduct(R2).transpose();
                TT2.row(count) = TT2.row(count - 1) + R2.cwiseProduct(R2).transpose() * CRN::h;
                // cout <<"BM"<< endl;
                // bm1 << BM1.row(count) << endl;
                // bm2 << BM1.row(count) << endl;
            }

            else
            {
                // count_out++;
                cout <<"restart ! "<< endl;

                cout << "X1 = " << X1.transpose() << endl;
                cout << "X2 = " << X2.transpose() << endl;
                // system("pause");
                return -1;
                // continue;
            }

            // Diffusion trajectories
            // traj1 << X1.transpose() << endl;
            // traj2 << X2.transpose() << endl;
        }
        total_length_BM = count + 1;
        return count * CRN::h;
    }
    else
    {
        cout << "X1 = " << X1.transpose() << endl;
        cout << "X2 = " << X2.transpose() << endl;
        return -1;
    }
}

// drift Model (ode)
VectorXd ReflectionCoupling::drift(const VectorXd& X)
{
    VectorXd a = CRN::propensity_c(X);
    VectorXd drift = VectorXd::Zero(CRN::d); 
    for (int j = 0; j < CRN::k; ++j) {
        drift += CRN::stoich[j].cast<double>() * (a[j] / CRN::volume);
    }
    return drift;
}


// Reflection coupling
VectorXd ReflectionCoupling::reflection(const VectorXd &X1, const VectorXd &X2, const VectorXd &rnd1, MatrixXd &A)
{
    VectorXd e = A.inverse() * (X1 - X2);
    e /= e.norm();
    MatrixXd P = MatrixXd::Identity(CRN::d,CRN::d) - 2.0 * e * e.transpose();
    return P * rnd1;
}

// // Solving k W from d W
// VectorXd ReflectionCoupling::solveW(const MatrixXd &M, const VectorXd &tmp)
// {
//     return M.jacobiSvd(ComputeThinU | ComputeThinV).solve(tmp);
// }

VectorXd ReflectionCoupling::solveW(const MatrixXd &M, const VectorXd &tmp, const VectorXd &rnd_bar)
{
    // SVD decomposition
    auto svd = M.jacobiSvd(ComputeThinU | ComputeThinV);
    // Range space
    VectorXd W_range = svd.solve(tmp);
    // Null space
    VectorXd W_null = rnd_bar - svd.solve(M * rnd_bar);
    return W_range + W_null;
}


// M*M'=N
MatrixXd ReflectionCoupling::Cov(const VectorXd& X)
{
    MatrixXd m = M(X);
    return m * m.transpose();
}

// Square root matrix (generic for any dimension d, assuming A is symmetric positive definite)
MatrixXd ReflectionCoupling::new_sigma(const MatrixXd &A)
{
    SelfAdjointEigenSolver<MatrixXd> es(A);

// Get eigenvalues and take square root (element-wise)
    VectorXd eigenvalues = es.eigenvalues().cwiseSqrt(); 
// Diagonal matrix with sqrt eigenvalues
    MatrixXd D = eigenvalues.asDiagonal(); 
// sqrt(A) = U * sqrt(D) * U^T
    MatrixXd sqrtA = es.eigenvectors() * D * es.eigenvectors().transpose();

    return sqrtA;
}

// MatrixXd ReflectionCoupling::new_sigma(const MatrixXd &A)
// {
//     MatrixXd tmp(CRN::d,CRN::d);
//     double s = sqrt(A.determinant());
//     double t = sqrt(A.trace() + 2*s);
//     tmp = 1/t * (A + s * MatrixXd::Identity(CRN::d, CRN::d));
//     return tmp;
// }


// Build M matrix for Chemical Langvian equation (size S_num x rea_num)
MatrixXd ReflectionCoupling::M(const VectorXd &X)
{
    VectorXd a = CRN::propensity_c(X);
    MatrixXd M = MatrixXd::Zero(CRN::d, CRN::k);
    for (int j = 0; j < CRN::k; ++j) {
        double sqrt_a = sqrt(a[j]);
        // Cast integer stoich to double for proper arithmetic
        M.col(j) = CRN::stoich[j].cast<double>() * sqrt_a;
    }
    return M/CRN::volume;
}

// Square root rate
VectorXd ReflectionCoupling::sqrtrate(const VectorXd &X)
{
     VectorXd a = CRN::propensity_c(X);
    for (int j = 0; j < CRN::k; ++j) {
        a[j] = sqrt(a[j]);
    }
    return a;
}

// Checks whether all elements of a vector are positive
bool ReflectionCoupling::allPositive(const VectorXd& v) {
    return (v.array() > -1e-10).all();
}

// Convert matrices to arrays
void ReflectionCoupling::convert_to_vectors(vector<vector<double>>& BM1_vec, 
                                            vector<vector<double>>& TT1_vec,
                                            vector<vector<double>>& BM2_vec, 
                                            vector<vector<double>>& TT2_vec) const {
    
    // Resize vectors to match the matrices' dimensions (k columns and 'length' rows)
    BM1_vec.resize(CRN::k, vector<double>(total_length_BM));
    TT1_vec.resize(CRN::k, vector<double>(total_length_BM));
    BM2_vec.resize(CRN::k, vector<double>(total_length_BM));
    TT2_vec.resize(CRN::k, vector<double>(total_length_BM));

    // Populate the vectors with data from the input matrices
    for (int j = 0; j < CRN::k; ++j) {
        for (int i = 0; i < total_length_BM; ++i) {
            BM1_vec[j][i] = BM1(i, j);
            TT1_vec[j][i] = TT1(i, j);
            BM2_vec[j][i] = BM2(i, j);
            TT2_vec[j][i] = TT2(i, j);
        }
    }
}








