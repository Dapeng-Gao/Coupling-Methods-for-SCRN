// #define NDEBUG
// #define EIGEN_NO_DEBUG
#ifndef NDEBUG
#define NDEBUG
#endif

#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#define _USE_MATH_DEFINES

#include <iostream>
#include <random>
#include <omp.h>
#include <math.h>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <filesystem>

#include "CRN.h"
#include "Prob.h"
#include "Reflection_coupling.h"
#include "Poisson.h"
#include "Dictionary.h"


using namespace std;
using namespace Eigen;
namespace fs = std::filesystem;

// Poisson coupling
int main()
{   
// Clear all output files
    // ofstream("Poisson_traj1_sample1.txt").close();
    // ofstream("Poisson_traj2_sample1.txt").close();
    // ofstream("traj_X1.txt").close();
    // ofstream("traj_X2.txt").close();
    // ofstream("BM1.txt").close();
    // ofstream("BM2.txt").close();

    // cout << "begin" << endl;

    int N_sample = 1000; // sample size
    int N_thread = 10; // number of parallel threads

    // cout << "wrong" << endl;

    // Initialize reactions matrix and rates & d and k
    CRN::initialize("config/Reactions.txt", "config/Rates.txt", "config/Initial_values.txt");
    // CRN::initialize("SIR_reactions.txt", "SIR_rates.txt");
    // CRN::initialize("oregonator_reactions.txt", "oregonator_rates.txt");

    cout << "V = " << CRN::volume << endl;
    // Class
    CRN EXP;
    Dictionary dict;
    Poisson Pois;

    // Creating Dictionary file
    fs::create_directory("cache");
    const string dictFile = "cache/dict.bin";
    if (fs::exists(dictFile)) {
        cout << "exist: " << dictFile << "\n";
        dict.load(dictFile);
    }
    else {
        cout << "no file: " << dictFile << "\n";
        dict.initialize();
        dict.save(dictFile);
        cout << "Saved";
    }
    cout << "initialize completed" << endl;

    // Create output directory
    fs::create_directory("output");
    fs::current_path("output");

    vector<VectorXi> reaction_vectors = CRN::stoich; // Reaction vectors

    // coupling time
    VectorXd coupling_time(N_sample);

// Parallel computing
#pragma omp parallel num_threads(N_thread)
    {
        // Random variable
        int rank = omp_get_thread_num();
        // int size = omp_get_num_threads();

        // mt19937 mt(0 + rank);

        // random_device rd;
        // mt19937 mt(rd() + rank);
        // normal_distribution<double> normal(0.0, 1.0);

#pragma omp for

        for(int i = 0; i < N_sample; i++)
        {
            random_device rd;
            mt19937 mt(rd() + rank);
            // mt19937 mt(1000 + i);
            normal_distribution<double> nm(0.0, 1.0);
            uniform_real_distribution<double> u(0.0, 1.0);
        // Initial values
            VectorXd X1_start = CRN::X1_default;
            VectorXd X2_start = CRN::X2_default;
            // cout << X1_start << endl;
            // cout << X2_start << endl;
            //cout << " i = " << i << endl;
            ReflectionCoupling W;
        // initial value
            bool coupled = false;
            int out_of_bd_flag = 0;
            double total_time = 0;
            int s = 0;

            // int refcouple = 0;
            // int poiscouple = 0;

        // CTMC Coupling method
            while (!coupled && s < 1e6)
            {
                s++;
                // random_device rd;
                // mt19937 mt1(rd()), mt2(rd());

                // normal_distribution<double> nm1(0.0, 1.0), nm2(0.0, 1.0);
                // uniform_real_distribution<double> u1(0.0, 1.0), u2(0.0, 1.0);

                vector<vector<double>> BM1_vec, TT1_vec, BM2_vec, TT2_vec;

            // reflection coupling
                double tmp_time = W.simulate(X1_start, X2_start, mt, nm, u);
                // refcouple+=1;

                // cout << "simulate " << endl;
                // system("pause");

                if(tmp_time < 0)
                {
                    cout<<" out of boundary! ";
                    out_of_bd_flag = 1;
                    break;
                }
                //cout<<" tmp_time = "<<tmp_time<<endl;
                total_time += tmp_time;

                W.convert_to_vectors(BM1_vec, TT1_vec, BM2_vec, TT2_vec); // Convert matrices to arrays

                
            // KMT matching and CTMC
                // cout << "get p" << endl;
                VectorXd last_P1 = Pois.get_p(BM1_vec, TT1_vec, mt, nm, u, 1, X1_start, i);
                // cout << "get p1 completed" << endl;
                VectorXd last_P2 = Pois.get_p(BM2_vec, TT2_vec, mt, nm, u, 2, X2_start, i);
                // cout << "get p2 completed" << endl;
            // calculate diff and prob
                VectorXd diff = last_P1 - last_P2;

            // coupling area
                double epsilon = 0.0401;
            // radius
                if ((diff.array().abs() <= epsilon).all())
                {
                    // poiscouple+=1;
                    // ds1 << last_P1.transpose() << endl;
                    // ds2 << last_P2.transpose() << endl;
                    
                    VectorXi coeffs(2 * CRN::k);
                    // cout << "begin prob" << endl;
                    // system("pause");
                    double prob = dict.Prob(EXP.rate(last_P1), EXP.rate(last_P2), CRN::volume * diff, coeffs);
                    // Parallel output critical
                    #pragma omp critical
                    {
                    // key check
                        // cout << "Difference: " << diff.transpose() << ", Probability: " << prob << endl;
                        // system("pause");
                        // cout<<"coeffs "<< coeffs.transpose()<<endl;
                        // system("pause");
                    }

                // random number judgment
                    // random_device rd;
                    // mt19937 mt(rd());
                    // uniform_real_distribution<double> u(0.0, 1.0);
                    double p = u(mt);
             
                // maximal coupling of poisson
                    if (p <= prob)
                    {
                        total_time += EXP.h;
                        coupled = true;
                        VectorXi coeff(CRN::k);
                        VectorXd tmp_last_p1 = last_P1;
                        EXP.SSA(tmp_last_p1, EXP.h, coeff, mt, u);
                        VectorXd tmp_last_p2 = tmp_last_p1;

                        // Log the coupling information
                        // ofstream pf1("Poisson_traj1_sample1.txt", ios::app);
                        // ofstream pf2("Poisson_traj2_sample1.txt", ios::app);
                        // pf1 << tmp_last_p1.transpose() << endl;
                        // pf2 << tmp_last_p2.transpose() << endl;
                        
                        coupling_time(i) = total_time;
                        
                        cout << "Coupled at time " << total_time << endl;
                        // cout << refcouple << endl;
                        // cout << poiscouple << endl;
                        break;
                    }

                // false and resample
                    else
                    {

                        total_time += EXP.h;
                        while(1)
                        {
                            //cout<<" conditioning "<<endl;
                            VectorXd tmp_last_p1 = last_P1;
                            VectorXd tmp_last_p2 = last_P2;
                            VectorXi coeff1(CRN::k), coeff2(CRN::k);
                            EXP.SSA(tmp_last_p1, EXP.h, coeff1, mt, u);
                            EXP.SSA(tmp_last_p2, EXP.h, coeff2, mt, u);

                            int sum = 0;
                            int num = 2 * CRN::k;
                            for(int i = 0; i < coeffs.size()/num; i++)
                            {
                                int local_sum = 0;
                                for(int j = 0; j < CRN::k; j++)
                                {
                                    if( coeff1(j) == coeffs(i*num+j) )
                                        local_sum++;
                                }
                                for(int j = 0; j < CRN::k; j++)
                                {
                                    if( coeff2(j) == coeffs(i*num+ CRN::k + j) )
                                        local_sum++;
                                }
                                if( local_sum == num)
                                {
                                    sum ++;
                                }
                            }
                            if(sum == 0)
                            {
                                last_P1 = tmp_last_p1;
                                last_P2 = tmp_last_p2;
                                break;
                            }
                        }
                        X1_start = last_P1;
                        X2_start = last_P2;
                    }
                }
                else
                {
                    // failed, update the initial value
                    X1_start = last_P1;
                    X2_start = last_P2;
                }
            }
            if(out_of_bd_flag == 1)
            {
                continue;
            }
        }
    }

    // Coupling time distribution
    sort(coupling_time.data(), coupling_time.data() + coupling_time.size());
    double ddx = 0.02;
    int MAX = floor(coupling_time.maxCoeff()/ddx) + 1;
    VectorXi distribution(MAX + 1);
    distribution(0) = N_sample;
    int time_count = 0;

    for(int i = 0; i < MAX; i++)
    {
        while(floor(coupling_time(time_count)/ddx) == i)
        {
            time_count++;
        }
        distribution(i+1) = N_sample - time_count;
    }

    // save coupling distribution
    ofstream myfile, myfile1;
    myfile.open("coupling.txt");
    myfile1.open("coupling_dist.txt");
    myfile << coupling_time << endl;
    myfile1 << distribution << endl;
    cout << "Coupling time is generated" << endl;

    return 0;
}
