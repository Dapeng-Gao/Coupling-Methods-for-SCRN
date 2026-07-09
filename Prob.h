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
    static double G_dist(int x, double t, double delta) {
        double N1 = 0;
        double a = 0;
        for (double i = 0; i < 10000 ; i = i + 1) {
            a = poissonCDF(i, pow(2, x) * delta); 
            if (a > t && i > 0) {
                N1 = i - 1;
                break;
            }
        }
        return N1;
    }

    // Probability
    static double prob(double k, double lambda)
    {
        double result = 0;
        result = exp(k*log(lambda) - lambda - lgammaf(k+1));
        return result;
    }   

    // Conditional Probability
    static double cond_prob(double t, double j, double lambda) {
        double result = 0;
        if (t < -j) {
            result = 0;
        } else if (t > j) {
            result = 1;
        } else {
            for (int i = -j; i < t; i = i + 2) {
                result += (prob((j - i) / 2, lambda) * prob((j + i) / 2, lambda)) / prob(j, 2 * lambda);
            }
        }
        return result;
    }

    // Conditional Distribution
    static double cond_dist(int q, double t, double y, double delta) {
        double N1 = 0;
        double temp = sqrt(delta) * y + pow(2, q) * delta;
        if (temp <= 0) {
            N1 = 0;
        } else {
            for (int i = -temp; i <= temp; i = i + 1) {
                double a = cond_prob(i, temp, pow(2, (q - 1)) * delta);
                N1 = i;
                if (a > t) {
                    N1 = (i - 1) / sqrt(delta);
                    return N1;
                    break;
                }
            }
        }
        return N1;
    }

private:

    

};

#endif
