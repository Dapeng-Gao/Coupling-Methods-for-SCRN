#ifndef CRN_H
#define CRN_H

#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>



using namespace std;
using namespace Eigen;

// REACTION NETWORK

class CRN {
    public:
    static VectorXd lambda; // parameter vector
    static int volume; // population size
    static double h; // time-step for τ-leap / CLE / Maximal coupling
    static void initialize(const string& rea_matrix, const string& rates, const string& ini); // Initialize reactions matrix and rates & d and k

    // initial values
    static VectorXd X1_default;
    static VectorXd X2_default;

    // Oregonator
    static int d; // Species
    static int k; // reaction

    // Reaction consumption and generation
    struct Reaction {
      VectorXi nu_react;   // S⁻
      VectorXi nu_prod;    // S⁺
    };

    // Reactions
    static vector<Reaction> reactions;

    // stoich
    static vector<VectorXi> stoich;   // ΔX = S⁺ - S⁻

    // propensities
    static VectorXd propensity_c(const VectorXd &X);  // Continuous propensities
    static VectorXd rate(const VectorXd &X); // rate = Continuous propensity * h

    // Gillespie / Stochastic Simulate Algorithm
    void SSA(VectorXd& P, double reaction_time, VectorXi& coeffs, mt19937& mt, uniform_real_distribution<double>& u);

    private:


};

#endif
