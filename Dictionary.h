#ifndef DICTIONARY_HPP
#define DICTIONARY_HPP

#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

using namespace std;
using namespace Eigen;

class Dictionary {
public:
    static constexpr int RADIUS = 5; // radius of difference // Oregonator -> 2
    static constexpr int BASE   = 2 * RADIUS + 1; // number of values for one dim
    static constexpr int MAXCOUNT = 4; // radius of coefficients // Oregonator -> 3

    // save and load dictionary
    void save(const string& filename) const;
    void load(const string& filename);


    void initialize(); // Initialize dictionary
    vector<VectorXi> Res_Area(int radius = RADIUS) const; // restrict area
    VectorXi LinearCombination(const VectorXi& a) const; // linearcombination a1R1 + ... + akRk
    void Combinations(int maxCount = MAXCOUNT); // Generate all possible linear combinations
    int findPairs(const VectorXi& diff, VectorXi& list_of_coeff); // Diff -> (a11,...,a1k, a21,...,a2k) -> count -> dictionary

    double Prob(const VectorXd& rates1, const VectorXd& rates2, const VectorXd& diff, VectorXi& coeffs_output) const; // couple prob of diff
    
private:
    // compare vectors
    struct CompareVectorXi {
        bool operator()(const VectorXi& a, const VectorXi& b) const;
    };

    multimap<VectorXi, VectorXi, CompareVectorXi> mapping; // lin - combo
    multimap<VectorXi, VectorXi, CompareVectorXi> diff_mapping; // diff - combo

    vector<VectorXi> dictionary; // dictionary[count] -> pairs
    
    // Generate factorial cache
    int Factorial(int n) const;
    static vector<int> factorial_cache;
};

#endif
