#ifndef REFLECTION_COUPLING_H
#define REFLECTION_COUPLING_H

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

#include "CRN.h"


using namespace std;
using namespace Eigen;

class ReflectionCoupling {

    public:
    ReflectionCoupling();

    VectorXd drift(const VectorXd& X); // drift (ode)
    double simulate(const VectorXd& X1_init, const VectorXd& X2_init, mt19937 &mt, normal_distribution<double> &nm, 
                        uniform_real_distribution<double> &u); // Reflection coupling of Diffusion approximation
    void convert_to_vectors(vector<vector<double>>& BM1_vec, vector<vector<double>>& TT1_vec, // Convert matrices to arrays
                            vector<vector<double>>& BM2_vec, vector<vector<double>>& TT2_vec) const;

    // Approximating diffusion
    MatrixXd Cov(const VectorXd& X);
    MatrixXd new_sigma(const MatrixXd& A);
    MatrixXd M(const VectorXd& X);
    VectorXd sqrtrate(const VectorXd& X);
    // Checks whether all elements of a vector are positive
    bool allPositive(const VectorXd& v);    

    private:

    MatrixXd BM1, TT1, BM2, TT2;
    
    int N_res = 1e6; //reserved BM length
    int total_length_BM;

    // vector<int> syn_time;
    
    VectorXd reflection(const VectorXd &X1, const VectorXd &X2, const VectorXd &rnd1, MatrixXd &A); // Reflection coupling
    VectorXd solveW(const MatrixXd& M, const VectorXd& tmp, const VectorXd &rnd_bar); // Solving k W from d W
    VectorXd sampleFullGaussian(const MatrixXd& M, const VectorXd& tmp);
    VectorXd addNullSpaceNoise(const MatrixXd& M, const VectorXd& W_min, double h, mt19937& mt);
};

#endif
