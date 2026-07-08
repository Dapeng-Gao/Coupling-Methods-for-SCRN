#include "Dictionary.h"
#include "CRN.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>


// Generate points within a restricted area : colume-major order
vector<VectorXi> Dictionary::Res_Area(int radius) const {

    long total = 1;
    for (int i = 0; i < CRN::d; ++i) total *= BASE;   // total points

    vector<VectorXi> points;
    points.reserve(total); // Allocate capacity in advance to speed up push_back

    vector<int> idx(CRN::d, 0); // size=d, all 0

    for (long count = 0; count < total; ++count) {
        // Convert idx to actual coordinate vector
        VectorXi pt(CRN::d);
        for (int k = 0; k < CRN::d; ++k) {
            pt[k] = idx[k] - radius;  // idx[k]=0 -> -radius，…，idx[k]=base-1 -> +radius
        }
        points.push_back(pt);

        // Auto-increasing binary index
        for (int k = CRN::d - 1; k >= 0; --k) {
            if (++idx[k] < BASE) {
                break;    // This bit is not full yet, continue to the next main loop
            }
            idx[k] = 0;  // This bit wraps around and moves to a higher bit.
        }
    }

    return points;
}

// Compute linear combinations
VectorXi Dictionary::LinearCombination(const VectorXi& a) const {
    VectorXi result = VectorXi::Zero(CRN::d);
    for (int i = 0; i < CRN::k; ++i) {
        result += a[i] * CRN::stoich[i];

    }
    return result;
}

// Generate all possible linear combinations
void Dictionary::Combinations(int maxCount) {
    mapping.clear();
    VectorXi combo = VectorXi::Zero(CRN::k);

    function<void(int)> dfs = [&](int idx) {
        if (idx == CRN::k) {
            VectorXi lin = LinearCombination(combo);
            mapping.emplace(lin, combo); // diff -> multiple combinations
            return;
        }
        for (int c = 0; c <= maxCount; ++c) {
            combo[idx] = c;
            dfs(idx + 1);
        }
    };
    dfs(0);

}

// Diff -> (a11,...,a1k, a21,...,a2k) -> count -> dictionary
int Dictionary::findPairs(const VectorXi& diff, VectorXi& list_of_coeff) {
    
    int count = 0;
    const int block = 2 * CRN::k;

    // 1) Check the cache
    auto dr = diff_mapping.equal_range(diff);
    if (dr.first != dr.second) {
        for (auto it = dr.first; it != dr.second; ++it) {
            const VectorXi& coeffs = it->second;
            VectorXi a = coeffs.head(CRN::k);
            VectorXi b = coeffs.tail(CRN::k);
            
            VectorXi f1 = LinearCombination(a);
            VectorXi f2 = LinearCombination(b);
            VectorXi calc_diff = f1 - f2;
            
            list_of_coeff.segment(count * block, block) = coeffs;
            ++count;
        }
        return count;
    }

    // 2) Cache miss: compute from mapping
    for (const auto& [f1, comb1] : mapping) {
        VectorXi target = f1 - diff;
        
        auto range = mapping.equal_range(target);
        int found = distance(range.first, range.second);
        if (found == 0) continue;
        
        for (auto jt = range.first; jt != range.second; ++jt) {
            const VectorXi& comb2 = jt->second;
            VectorXi f2 = LinearCombination(comb2);
            VectorXi calc_diff = f1 - f2;

            // Merge two combinations comb1, comb2
            VectorXi combined(block);
            combined.head(CRN::k) = comb1;
            combined.tail(CRN::k) = comb2;

            // Cache this new result
            diff_mapping.emplace(diff, combined);

            // Also write to the output list
            list_of_coeff.segment(count * block, block) = combined;
            ++count;
        }
    }
    return count;
}

// Initialize the dictionary
void Dictionary::initialize() {

    int maxCombos = 1;
    for (int i = 0; i < CRN::k; ++i) maxCombos *= (MAXCOUNT+1);
    int maxLen = maxCombos * maxCombos * 2 * CRN::k;
    Combinations(); // construct mapping

    // 2. generate points
    auto points = Res_Area();
    dictionary.reserve(points.size());

    for (const auto& point : points) {
        VectorXi tmp(maxLen);
        int count = findPairs(point, tmp);
        tmp.conservativeResize(count * 2 * CRN::k);
        dictionary.push_back(tmp);
    }

}

// couple prob of diff
double Dictionary::Prob(const VectorXd& rates1,
                        const VectorXd& rates2,
                        const VectorXd& diff,
                        VectorXi& coeffs_output) const {
    // 1) First round diff and add RADIUS to get the coordinates on [0, BASE)
    vector<int> idx(CRN::d);
    for (int i = 0; i < CRN::d; ++i) {
        int v = round(diff(i)) + RADIUS;
        if (v < 0 || v >= BASE) return 0.0;
        idx[i] = v;
    }

    // 2) Multidimensional coordinate flattening: index = idx[0]*BASE^(d-1) + idx[1]*BASE^(d-2) + … + idx[d-1]
    size_t index = 0;
    for (int i = 0; i < CRN::d; ++i) {
        index = index * BASE + idx[i];
    }

    // 3) Take out the corresponding coefficient vector
    VectorXi coeffs = dictionary[index];
    coeffs_output = coeffs;

    // 4) Count how many pairs (comb1, comb2) there are: each pair of coefficients has a length of 2*k
    int pair_len = 2 * CRN::k;
    int num_pairs = coeffs.size() / pair_len;

    double prob = 0.0;
    double p = 0.0;   
    // 5) Accumulate Poisson probability for each pair
    for (int i = 0; i < num_pairs; ++i) {
        double p1 = 1.0, p2 = 1.0;
        for (int j = 0; j < CRN::k; ++j) {
            int c1 = coeffs(i * pair_len + j);
            int c2 = coeffs(i * pair_len + CRN::k + j);
            // prob compute
            p1 *= exp(-rates1(j)) * pow(rates1(j), c1) / Factorial(c1);
            p2 *= exp(-rates2(j)) * pow(rates2(j), c2) / Factorial(c2);
        }
        prob += p1 * p2;
        // p += min(p1, p2);
        // prob = min(1.0, p);

    }
    return prob;
}

vector<int> Dictionary::factorial_cache = {1, 1, 2, 6, 24}; // Pre-catch small values
// Generate factorial cache
int Dictionary::Factorial(int n) const {
    // Make sure the factorial cache is large enough
    if (n >= factorial_cache.size()) {
        for (int i = factorial_cache.size(); i <= n; ++i) {
            factorial_cache.push_back(factorial_cache.back() * i);
        }
    }
    return factorial_cache[n];
}

// compare vectors
bool Dictionary::CompareVectorXi::operator()(const VectorXi& a, const VectorXi& b) const {
    const int n = a.size();
    for (int i = 0; i < n; ++i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false;
}

// save to file
void Dictionary::save(const string& filename) const {
    ofstream out(filename, ios::binary);
    if (!out) throw runtime_error("Cannot open file to save dictionary");

    size_t dict_size = dictionary.size();
    out.write(reinterpret_cast<const char*>(&dict_size), sizeof(dict_size));

    for (const auto& vec : dictionary) {
        int len = vec.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(reinterpret_cast<const char*>(vec.data()), len * sizeof(int));
    }
    out.close();
}

// load file
void Dictionary::load(const string& filename) {
    ifstream in(filename, ios::binary);
    if (!in) throw runtime_error("Cannot open file to load dictionary");

    size_t dict_size;
    in.read(reinterpret_cast<char*>(&dict_size), sizeof(dict_size));
    dictionary.resize(dict_size);

    for (size_t i = 0; i < dict_size; ++i) {
        int len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        dictionary[i].resize(len);
        in.read(reinterpret_cast<char*>(dictionary[i].data()), len * sizeof(int));
    }
    in.close();
}
