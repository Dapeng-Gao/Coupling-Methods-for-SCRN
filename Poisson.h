#ifndef Poisson_H
#define Poisson_H

#include <iostream>
#include <random>
#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <sys/time.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

#include "Prob.h"


using namespace std;
using namespace Eigen;

class Poisson {
    
public:

    // KMT matching
    void KMT(vector<double> &W, vector<double> &P, int Np, mt19937 &mt, normal_distribution<double> &norm);

    // REACTION NETWORK JUMP
    VectorXd state_increment(const vector<vector<double>>& P, const VectorXi& index_old, const VectorXi& index_new); // CTMC for the SCRN

    // Get poisson process(ctmc) (matching diffusion)
    MatrixXd get_p(const vector<vector<double>> &BM, const vector<vector<double>> &TT, mt19937 &mt, 
                   normal_distribution<double> &norm, uniform_real_distribution<double> &u, 
                   int trajectory_index, const VectorXd& X_start, int sample_index);

    // Brownian Bridge
    double BBI(double t, double t1, double BM1, double t2, double BM2, mt19937 &mt, normal_distribution<double> &nm);

    // Brownian Bridge Interpolation
    void INPTBBI(const vector<vector<double>>& TT, const vector<vector<double>>& BM, vector<vector<double>>& i_TT, 
                 vector<vector<double>>& i_BM, mt19937 &mt, normal_distribution<double> &nm);

    // Index floor
    VectorXi FLOOR(const VectorXd index);

    // Next power of 2 utility function 
    int nextPowerOf2(int x);

private:
    // KMT matching step size
    double delta = 0.01;


};

#endif
