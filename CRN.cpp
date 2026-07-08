#include "CRN.h"

#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

using namespace std;

// Species and reactions
int CRN::volume = 100; // population size
double CRN::h = 0.0001; // Poisson time size & diffusion time size & Maximal coupling step size
int CRN::d = 0; // Species
int CRN::k = 0; // ReactionS
VectorXd CRN::X1_default; // Initial values
VectorXd CRN::X2_default;
VectorXd CRN::lambda; // Reaction rates
vector<VectorXi> CRN::stoich;
vector<CRN::Reaction> CRN::reactions; // Reactions matrix


// Initialize reactions matrix and rates & d and k 
void CRN::initialize(const string& rea_matrix, const string& rates, const string& ini) {
    ifstream reactions_in(rea_matrix);

    // read all colume，give d and k
    vector<vector<int>> matrix_data;
    string line;
    while (getline(reactions_in, line)) {
        stringstream ss(line);
        vector<int> row;
        int val;
        while (ss >> val) {
            row.push_back(val);
        }
        if (!row.empty()) {
            matrix_data.push_back(row);
        }
    }
    reactions_in.close();

    k = matrix_data.size();  // reaction number
    d = matrix_data[0].size() / 2;  // species number

    // reactions -> product
    reactions.clear();
    reactions.reserve(k);
    for (int j = 0; j < k; ++j) {
        Reaction r;
        r.nu_react = VectorXi(d);
        r.nu_prod = VectorXi(d);
        for (int i = 0; i < d; ++i) {
            r.nu_react(i) = matrix_data[j][i];
        }
        for (int i = 0; i < d; ++i) {
            r.nu_prod(i) = matrix_data[j][d + i];
        }
        reactions.push_back(r);
    }

    // stoich = nu_prod - nu_react
    stoich.clear();
    stoich.reserve(k);
    for (const auto& r : reactions) {
        stoich.push_back(r.nu_prod - r.nu_react);
    }

    // param lambda
    ifstream rates_in(rates);

    lambda.resize(k);
    for (int j = 0; j < k; ++j) {
        rates_in >> lambda(j);
    }
    rates_in.close();

    // Initial values
    ifstream ini_in(ini);
    if (!ini_in.is_open()) {
        cerr << "Error: Cannot open initial values file: " << ini << endl;
        return;
    }

    X1_default = VectorXd(d);
    X2_default = VectorXd(d);

    // X1_default
    for (int i = 0; i < d; ++i) {
        if (!(ini_in >> X1_default(i))) {
            cerr << "Error: Failed to read initial values for X1_default" << endl;
            return;
        }
    }

    // X2_default
    for (int i = 0; i < d; ++i) {
        if (!(ini_in >> X2_default(i))) {
            cerr << "Error: Failed to read initial values for X2_default" << endl;
            return;
        }
    }
    ini_in.close();

    cout << "Initialized CRN: d=" << d << ", k=" << k << endl;
    cout << "reactions:" << endl;
    for (const auto& r : reactions) {
        cout << "nu_react: " << r.nu_react.transpose() << ", nu_prod: " << r.nu_prod.transpose() << endl;
    }
    cout << "stoich: " << endl;
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < k; ++j) {
            cout << stoich[j](i) << " ";
        }
        cout << endl;
    }
    cout << "lambda: " << lambda.transpose() << endl;
    cout << "X1_default: " << X1_default.transpose() << endl;
    cout << "X2_default: " << X2_default.transpose() << endl;

}



// Continuous Propensity (Concentration / CLE)
VectorXd CRN::propensity_c(const VectorXd &X) {
    VectorXd Y(k);

    for (int i = 0; i < k; ++i) {
        const VectorXi &cons = reactions[i].nu_react;
        double a = lambda(i) * volume;

        for (int j = 0; j < d; ++j) {
            int c = cons(j);
            if (c == 0) continue;
            // a *= pow(X(j) / volume, c);  // (X/V)^c, X Poisson
            a *= pow(X(j), c);  // (X)^c, X Concentration
        }
        Y(i) = a;
    }
    return Y;
}

// rate = continuous Propensity * time step h
// Concentration X
VectorXd CRN::rate(const VectorXd &X) {
    return propensity_c(X) * h;
}

// Gillespie stochastic simulate algorithm
void CRN::SSA(VectorXd &P, double reaction_time, VectorXi& coeffs, mt19937& mt, uniform_real_distribution<double>& u)
{
    // random_device rd;
    // mt19937 mt(rd());
    // uniform_real_distribution<double> u1(0.0, 1.0), u2(0.0, 1.0);

    coeffs.setZero();
    // VectorXd PC = P * volume;

    double current_time = 0.0;

    while (current_time < reaction_time) {

        VectorXd rates = rate(P);
        double total_rate = rates.sum();
        
        if (total_rate <= 0.0) break;

        // Sample wating time τ ~ Exp(total_rate)
        double tau = -log(1.0 - u(mt)) / total_rate;
        if (current_time + tau > reaction_time) break;
        current_time += tau;
        
        double threshold = u(mt) * total_rate;
        double cumsum = 0.0;
        int reaction_index = 0;
  
        for (; reaction_index < k; ++reaction_index)
        {
            cumsum += rates[reaction_index];
            if (cumsum >= threshold || reaction_index == k - 1) break;
        }
        reaction_index = min(reaction_index, k-1);
        coeffs[reaction_index]++;

        // update
        P += stoich[reaction_index].cast<double>()/volume;
    }
}











