#include "Poisson.h"
#include "Prob.h"
#include "CRN.h"

#include <iostream>
#include <fstream>
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

// KMT algorthm for matching poisson and brownian motion trajectory
void Poisson::KMT(vector<double> &W, vector<double> &P, int Np, mt19937 &mt, normal_distribution<double> &norm)
{
    double sq_delta = sqrt(delta);
    int w_size = W.size();

    VectorXd T(Np + 1);
    double partial_sum = 0;

    for (int i = 1; i <= Np; ++i)
    { 
        if(i < w_size)
            partial_sum += (W[i] - W[i - 1]) / sq_delta;
        else
            partial_sum += norm(mt);
        T[i] = partial_sum;           
    }

    int q = round(log(Np)/log(2)) - 1;
    
    vector<double> p2(q + 3);
    vector<double> inv_sq2_p(q + 3);
    for (int i = 0; i < q + 3; ++i) {
        p2[i] = pow(2, i);
        inv_sq2_p[i] = pow(sqrt(2), -i);
    }

    vector<VectorXd> U(q + 2);
    for (int i = 0; i < q + 2; ++i) {
        U[i] = VectorXd::Zero(Np / p2[i]); 
    }

    for (int i = 0; i < q + 1; ++i) {
        double phi = Prob::normalCDF(inv_sq2_p[i] * (T[p2[i+1]] - T[p2[i]]));
        U[i][1] = (Prob::G_dist(i, phi, delta) - p2[i] * delta) / sq_delta;
    }
        
    vector<VectorXd> v_tuta(q + 1);
    for (int i = 1; i <= q; ++i) {
        v_tuta[i] = VectorXd::Zero(Np / p2[i]); 
    }

    for (int i = 1; i <= q; ++i) {
        int k = Np / p2[i] - 1;
        int half_step = p2[i - 1]; 
        for (int j = 1; j <= k; ++j) {
            int pos1 = (2 * j + 1) * half_step;
            int pos2 = (2 * j) * half_step;
            int pos3 = (2 * j + 2) * half_step;
            v_tuta[i][j] = 2 * T[pos1] - T[pos2] - T[pos3];
        }
    }
        
    double phi_init = Prob::normalCDF(T[1]);
    double N1_bar = (Prob::G_dist(0, phi_init, delta) - delta) / sq_delta; 

    vector<VectorXd> U_tuta(q + 1);
    for (int i = 1; i <= q; ++i) {
        U_tuta[i] = VectorXd::Zero(Np / p2[i]);
    }

    for (int i = 1; i <= q; ++i) {
        double phi = Prob::normalCDF(inv_sq2_p[i] * v_tuta[i][1]);
        U_tuta[i][1] = Prob::cond_dist(i, phi, U[i][1], delta);
    }

    int c1 = 1;
    int c2 = 1;

    for (int u = 1; u <= q; ++u)
    {
        for (int v = 1; v <= c2; ++v)
        {
            int d2 = 2 * c1;
            int d3 = 2 * c1 + 1;
            int loop_end = q + 1 - u;

            for (int j = 1; j <= loop_end; ++j)
            {   
                int d1 = j - 1;
                U[d1][d2] = (U[j][c1] + U_tuta[j][c1]) * 0.5;
                U[d1][d3] = (U[j][c1] - U_tuta[j][c1]) * 0.5;
            }

            for (int p = 2; p <= loop_end; ++p)
            {
                int d1 = p - 1;
                double phi_1 = Prob::normalCDF(inv_sq2_p[d1] * v_tuta[d1][d2]);
                double phi_2 = Prob::normalCDF(inv_sq2_p[d1] * v_tuta[d1][d3]);
                U_tuta[d1][d2] = Prob::cond_dist(d1, phi_1, U[d1][d2], delta);
                U_tuta[d1][d3] = Prob::cond_dist(d1, phi_2, U[d1][d3], delta);
            }
            c1 += 1;
        }
        c2 <<= 1; 
    }

    double current_NN = N1_bar * sq_delta + delta;
    P[0] = std::ceil(current_NN);

    for (int i = 1; i < Np; ++i)
    {
        current_NN += U[0][i] * sq_delta + delta; 
        P[i] = std::ceil(current_NN);
    }
    
    P.resize(w_size);
}

// CTMC for the SCRN
VectorXd Poisson::state_increment(const vector<vector<double>> &P, const VectorXi& index_old, const VectorXi& index_new)
{
    VectorXd delta_X = VectorXd::Zero(CRN::d);
        for (int k = 0; k < CRN::k; k++) {
            double dP = P[k][index_new(k)] - P[k][index_old(k)];
            delta_X += CRN::stoich[k].cast<double>() * dP;
        }
        return delta_X;
}

// Get poisson process(ctmc) (matching diffusion)
MatrixXd Poisson::get_p(const vector<vector<double>> &BM, const vector<vector<double>> &TT, mt19937 &mt, normal_distribution<double> &norm, 
                        uniform_real_distribution<double> &u, int trajectory_index, const VectorXd& X_start, int sample_index)
{
    // string filename = "Poisson_traj" + to_string(trajectory_index) 
    //                + "_sample" + to_string(sample_index + 1) + ".txt";
    // ofstream outfile(filename, ios::app);
    
    // initial values
    // outfile << X_start.transpose() << endl;

    // cout << "begin getp" << endl;

    // cout<<"inside get p function, X_start = "<<X_start.transpose()<<endl; 
    vector<vector<double>> i_TT(TT.size());
    vector<vector<double>> i_BM(BM.size());

    /*
    for (size_t k = 0; k < BM.size(); ++k) {
        cout << "Size of BM[" << k << "]: " << BM[k].size() << endl;
    }

    for (size_t k = 0; k < TT.size(); ++k) {
        cout << "Size of TT[" << k << "]: " << TT[k].size() << endl;
    }
    */
     
    // Brownian bridge interpolation
    // cout << "interpolation" << endl;
    // system("pause");
    INPTBBI(TT, BM, i_TT, i_BM, mt, norm);
    // cout<<"BBI done"<<endl;
    /*
    for(int k = 0; k < 4; k++)
    {
        cout<<"i_BM["<<k<<"] = ";
        for(int j = 0; j < i_BM[k].size(); j++)
        {
            cout<<i_BM[k][j]<<" ";
        }
        cout<<endl;
    }
    */

    vector<size_t> i_BM_lengths;
    i_BM_lengths.reserve(i_BM.size());
    for (size_t i = 0; i < i_BM.size(); ++i) {
        i_BM_lengths.push_back(i_BM[i].size());
    }

    vector<int> Np;
    for (int length : i_BM_lengths)
    {
        Np.push_back(nextPowerOf2(length));
    }
    int max_Np = *max_element(Np.begin(), Np.end());
    vector<vector<double>> P(i_BM.size(), vector<double>(max_Np));

    // cout << "i_BM size = " << i_BM.size() << endl;
    // KMT matching BM & Poss
    // cout << "before KMT" << endl;
    for (size_t k = 0; k < i_BM.size(); ++k)
    {
        // cout<<"KMP algorithm, k = "<<k<<endl;
        P[k].resize(2 * nextPowerOf2(i_BM[k].size()));
        // cout<<"i_BM size = "<<i_BM[k].size()<<" P size = "<<nextPowerOf2(2*i_BM[k].size())<<endl;
        // cout<<"Np = "<<P[k].size()<<endl;
        KMT(i_BM[k], P[k], P[k].size(), mt, norm);
        // cout<<" got P seq"<<endl;
        // system("pause");
        
    }
    // cout<<" KMT "<<endl;
    // system("pause");

    VectorXd index_X_start = VectorXd::Zero(CRN::k);
    VectorXd X_old = X_start;
    VectorXd X_new(CRN::d); // Poisson
    VectorXd index_X_old = index_X_start; // index initialize
    VectorXd index_X_new(CRN::k);
    VectorXi floor_X_old = FLOOR(index_X_old);
    VectorXi floor_X_new(CRN::k);
    int flag = 0;
    int k = 0;
    while(k < 1e5)
    {
        k++;
        index_X_new = index_X_old + CRN::rate(X_old);
        floor_X_new = FLOOR(index_X_new);
        //cout<<"step "<<k<<endl;
        //cout<<index_X_old.transpose()<<endl;
        //cout<<index_X_new.transpose()<<endl;
        //cout<<floor_X_new<<endl;
        for(int i = 0; i < P.size(); i++)
        {
            //cout<<i_BM[i].size()<<" , ";
            if(floor_X_new(i) > i_BM[i].size())
            {
                flag ++;
            }
        }

        if(flag != 0)
        {
            break;
        }

        // cout << "run P" << endl;
        // system("pause");
        X_new = X_old +  state_increment(P, floor_X_old, floor_X_new)*1.0/ CRN::volume;
        //cout<<"X_new = "<<X_new.transpose()<<endl;

        // outfile << X_new.transpose() << endl;

        X_old = X_new;
        index_X_old = index_X_new;
        floor_X_old = floor_X_new;
    }
    // cout << "get p" << endl;
    return X_new;
}

// Brownian Bridge
double Poisson::BBI(double t, double t1, double BM1, double t2, double BM2, mt19937 &mt, normal_distribution<double> &nm) {
    double linear_part = BM1 + (t - t1) / (t2 - t1) * (BM2 - BM1);
    double variance_part = sqrt((t - t1) * (t2 - t) / (t2 - t1));
    return linear_part + variance_part * nm(mt);
}

// Brownian Bridge Interpolation
void Poisson::INPTBBI(const vector<vector<double>>& TT, const vector<vector<double>>& BM, vector<vector<double>>& i_TT, vector<vector<double>>& i_BM, mt19937 &mt, normal_distribution<double> &nm) {
    for (size_t k = 0; k < TT.size(); ++k) {
        double t_prev = TT[k][0];
        i_TT[k].push_back(t_prev);  // Include t1 (the first endpoint)
        i_BM[k].push_back(BM[k][0]);  // Include the corresponding BM1 value for t1

        for(size_t j = 1; j < TT[k].size(); j++) {
            double t1 = t_prev;
            double t2 = TT[k][j];

            // Interpolating between t1 and t2
            for (double t = i_TT[k].back() + delta; t < t2; t += delta) {
                i_BM[k].push_back(BBI(t, t1, i_BM[k].back(), t2, BM[k][j], mt, nm));
                i_TT[k].push_back(t);
            }
            if(t1 + delta < t2) // Move to the next interval
            {
                t_prev = t2;
            }
        }
    }
}

// Index floor
VectorXi Poisson::FLOOR(const VectorXd index)
{
    VectorXi tmp(CRN::k);
    for(int i=0; i < CRN::k; i++)
    {
        tmp(i) = floor(index(i)/delta);
    }

    return tmp;
}

// Next power of 2 utility function
int Poisson::nextPowerOf2(int x) {
    if (x <= 0) return 1;
    int power = 1;
    while (power < x) {
        power <<= 1;
    }
    return power;
}