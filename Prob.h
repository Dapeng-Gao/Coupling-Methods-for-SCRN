#ifndef PROB_HPP
#define PROB_HPP

#define _USE_MATH_DEFINES

#include <cmath>
#include <math.h>
#include <Eigen/Dense>
// #include "CRN.h"

using namespace Eigen;

class Prob {
public:

    // Normal CDF
    static double normalCDF(double x) {
        return erfc(-x / sqrt(2)) / 2;  // Phi(-∞, x) aka N(x)
    }

    // Poisson CDF
    static double poissonCDF(double x, double lambda) {
        double F = exp(-lambda);
        if (x > 0) {
            for (int i = 1; i < x; ++i) {
                F += exp(i * log(lambda) - lambda - lgammaf(i + 1));
            }
        }
        return F;
    }

    // Normal PDF
    static double normpdf(const VectorXd &X, const VectorXd &mu, MatrixXd &A) {
        VectorXd tmp = X - mu;
        return exp(-0.5 * tmp.transpose() * A.inverse() * tmp) / (2.0 * M_PI * sqrt(A.determinant()));
    }
    // static double normpdf(const VectorXd &X,
    //                   const VectorXd &mu,
    //                   const MatrixXd &A)
    // {
    //     int d = X.size();
    //     VectorXd diff = X - mu;
    //     VectorXd sol = A.ldlt().solve(diff);
    //     double quad = diff.dot(sol);
    //     double det = A.determinant();
    //     double norm = 1.0 / (pow(2.0 * M_PI, d / 2.0) * sqrt(det));
    //     return norm * exp(-0.5 * quad);
    // }

    // Compute G_j
    static double G_dist(int level, double t, double delta) {
        double lambda = static_cast<double>(1 << level) * delta;
        double F = std::exp(-lambda); 

        if (F > t) {
            return 0.0;
        }

        for (int i = 2; i < 10000; ++i) {
            F += std::exp((i - 1) * std::log(lambda) - lambda - std::lgammaf(static_cast<float>(i)));
            
            if (F > t) {
                return static_cast<double>(i - 1);
            }
        }
        return 9999.0;
    }

    // Probability
    static double prob(double k, double lambda) {
        return std::exp(k * std::log(lambda) - lambda - std::lgammaf(static_cast<float>(k + 1.0)));
    }  

    // Conditional Probability
    static double cond_prob(double t, double j, double lambda) {
        if (t < -j) return 0.0;
        if (t > j)  return 1.0;

        double denom = prob(j, 2.0 * lambda);
        double result = 0.0;
        
        int i_start = -static_cast<int>(j);
        for (int i = i_start; i < t; i += 2) {
            result += (prob((j - i) / 2.0, lambda) * prob((j + i) / 2.0, lambda)) / denom;
        }
        return result;
    }

    // Conditional Distribution
static double cond_dist(int q, double t, double y, double delta) {
        double sqrt_delta = std::sqrt(delta);
        double temp = sqrt_delta * y + static_cast<double>(1 << q) * delta;

        if (temp <= 0.0) return 0.0;

        double lambda = static_cast<double>(1 << (q - 1)) * delta;
        int i_start = static_cast<int>(-temp);
        int i_end   = static_cast<int>(temp);

        double N1 = 0.0;
        for (int i = i_start; i <= i_end; ++i) {
            double a = cond_prob(static_cast<double>(i), temp, lambda);
            N1 = i;
            if (a > t) {
                return (i - 1) / sqrt_delta;
            }
        }
        return N1; 
    }

private:

    

};

#endif
