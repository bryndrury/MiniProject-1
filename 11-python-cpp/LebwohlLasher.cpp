#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <chrono>
#include <mutex>
#include <omp.h>

// GSL for eigenvalues
#include <gsl/gsl_eigen.h>

void initdat(std::vector< std::vector< double > > &arr, int nmax, double scale) 
{
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        #pragma omp for collapse(2)
        for (int i = 0; i < nmax; i++)
        {
            for (int j = 0; j < nmax; j++)
            {
                arr[i][j] = dis(gen) * scale;
            }
        }
    }
}

double one_energy(const std::vector< std::vector< double > > &arr , int ix, int iy, int nmax) 
{
    double energy = 0.0;

    int ixp = (ix + 1) % nmax;          // These are the coordinates
    int ixm = (ix - 1 + nmax) % nmax;   // of the neighbours
    int iyp = (iy + 1) % nmax;          // with wraparound
    int iym = (iy - 1 + nmax) % nmax;   // (periodic boundary conditions implemented differently to the python)


    double ang ;

    ang = arr[ix][iy] - arr[ixp][iy];
    energy += 0.5*(1.0 - 3.0* pow(cos(ang), 2));

    ang = arr[ix][iy] - arr[ixm][iy];
    energy += 0.5*(1.0 - 3.0* pow(cos(ang), 2));

    ang = arr[ix][iy] - arr[ix][iyp];
    energy += 0.5*(1.0 - 3.0* pow(cos(ang), 2));

    ang = arr[ix][iy] - arr[ix][iym];
    energy += 0.5*(1.0 - 3.0* pow(cos(ang), 2));

    return energy;
}

double all_energy(const std::vector< std::vector< double > > &arr, int nmax) 
{
    double energy_all = 0.0; 

    #pragma omp parallel for reduction(+:energy_all)
    for (int i = 0; i < nmax; i++)
    {
        for (int j = 0; j < nmax; j++)
        {
            energy_all += one_energy(arr, i, j, nmax);
        }
    }

    return energy_all;
}

double get_order(const std::vector< std::vector< double > > &arr, int nmax) 
{
    std::vector< std::vector< double > > Qab(3, std::vector< double >(3, 0.0));
    // Equivalent to -> np.eye(3) in python
    std::vector< std::vector< double > > delta = { {1, 0 , 0}, {0, 1, 0}, {0, 0, 1} };;


    std::vector< std::vector< std::vector< double > > > lab(3, std::vector< std::vector< double > >(nmax, std::vector< double >(nmax , 0.0) ));

    // Equivalent to -> np.vstack([np.cos(arr), np.sin(arr), np.zeros_like(arr)]).reshape(3, nmax, nmax) in python
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nmax ; i++)
    {
        for (int j = 0; j < nmax; j++)
        {
            lab[0][i][j] = cos(arr[i][j]);
            lab[1][i][j] = sin(arr[i][j]);
            lab[2][i][j] = 0.0;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int a = 0; a < 3; a++)
    {
        for (int b = 0; b < 3; b++)
        {
            double temp_Qab = 0.0;

            #pragma omp parallel for collapse(2) reduction(+:temp_Qab)
            for (int i = 0; i < nmax; i++)
            {
                for (int j = 0; j < nmax; j++)
                {
                    temp_Qab += 3 * lab[a][i][j] * lab[b][i][j] - delta[a][b];
                }
            }
            Qab[a][b] = temp_Qab;
        }
    }

    double recip = (2 * nmax * nmax);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            Qab[i][j] /= recip;
        }
    }
    
    // Solve for eigenvalues using gsl
    gsl_matrix *QabMATRIX = gsl_matrix_alloc(3, 3);
    gsl_vector *eigen_values = gsl_vector_alloc(3);
    gsl_matrix *eigen_vectors = gsl_matrix_alloc(3, 3);
    gsl_eigen_symm_workspace *WORKSPACE = gsl_eigen_symm_alloc(3);

    for (int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            gsl_matrix_set(QabMATRIX, i, j, Qab[i][j]);
        }
    }

    gsl_eigen_symm(QabMATRIX, eigen_values, WORKSPACE);
    gsl_eigen_symmv_sort(eigen_values, eigen_vectors, GSL_EIGEN_SORT_VAL_DESC);

    double max_eigenvalue = gsl_vector_get(eigen_values, 0);

    gsl_eigen_symm_free(WORKSPACE);
    gsl_matrix_free(QabMATRIX);
    gsl_vector_free(eigen_values);
    gsl_matrix_free(eigen_vectors);

    return max_eigenvalue;
}

double MC_step(std::vector< std::vector< double > > &arr, double Ts, int nmax) 
{
    double scale = 0.1 + Ts;
    double accept = 0;
    
    std::vector< std::vector< double > > xran(nmax, std::vector< double >(nmax, 0.0));
    std::vector< std::vector< double > > yran(nmax, std::vector< double >(nmax, 0.0));
    std::vector< std::vector< double > > aran(nmax, std::vector< double >(nmax, 0.0));
    initdat(xran, nmax, nmax);
    initdat(yran, nmax, nmax);
    initdat(aran, nmax, scale);

    #pragma omp parallel reduction(+:accept)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        #pragma omp for collapse(2)
        for (int i = 0; i < nmax; i++)
        {
            for (int j = 0; j < nmax; j++)
            {
                double ix = xran[i][j];
                double iy = yran[i][j];
                double ang = aran[i][j];

                double en0 = one_energy(arr, ix, iy, nmax);

                double old_value = arr[ix][iy];
                arr[ix][iy] += ang;

                double en1 = one_energy(arr, ix, iy, nmax);

                if (en1 <= en0)
                {
                    accept +=1;
                }
                else
                {
                    double boltz = exp( -(en1 - en0) / Ts );

                    if (boltz >= (dis(gen)))
                    {
                        accept += 1;
                    }
                    else
                    {
                        arr[ix][iy] = old_value;
                    }
                }
            }
        }
    }
    return static_cast<double>(accept) / (nmax * nmax);
}

bool savedat(std::vector< std::vector< double > > &arr, int nsteps, double Ts, double runtime,
           std::vector<double> ratio, std::vector<double> energy, std::vector<double> order, int nmax) 
{
    auto time = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(time);
    std::tm *ltm = std::localtime(&current_time);

    char filename[100];
    std::strftime(filename, sizeof(filename), "LL-Output-%a-%d-%b-%Y-at-%I-%M-%S%p.txt", ltm);

    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not create file!" << std::endl;
        return false;
    }

    file << "#=====================================================" << std::endl;
    file << "# File created:        " << std::put_time(ltm, "%a-%d-%b-%Y-at-%I-%M-%S%p") << std::endl;
    file << "# Program:             CPP-Lebwohl-Lasher" << std::endl;
    file << "# Size of lattice:     " << nmax << "x" << nmax << std::endl;
    file << "# Number of MC steps:  " << nsteps << std::endl;
    file << "# Reduced temperature: " << std::fixed << std::setprecision(3) << Ts << std::endl;
    file << "# Number of threads:   " << omp_get_max_threads() << std::endl;
    file << "# Run time (s):        " << std::fixed << std::setprecision(6) << runtime << std::endl;
    file << "#=====================================================" << std::endl;
    file << "# MC step:  Ratio:     Energy:   Order:" << std::endl;
    file << "#=====================================================" << std::endl;

    for (int i = 0; i <= nsteps; ++i)
    {
        file << "   " << std::setw(5) << std::setfill('0') << i << "    "
                << std::fixed << std::setprecision(4) << ratio[i] << " "
                << std::fixed << std::setprecision(4) << energy[i] << "  "
                << std::fixed << std::setprecision(4) << order[i] << " " << std::endl;
    }

    file.close();
    return true;
}

bool LebwohlLasher(std::vector< std::vector< double > > &lattice, std::string program, int nsteps, int nmax, double temp, int threadcount) 
{
    omp_set_num_threads(threadcount);

    std::vector< double > energy(nsteps + 1, 0.0);
    std::vector< double > ratio(nsteps + 1, 0.0);
    std::vector< double > order(nsteps + 1, 0.0);
    
    energy[0] = all_energy(lattice, nmax);
    ratio[0] = 0.5;
    order[0] = get_order(lattice, nmax);

    auto initial_time = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter < nsteps+1; iter++)
    {
        std::cout << "Iter: " << iter << ", nsteps: " << ratio.size()-1 << "\r" << std::flush;
        ratio[iter] = MC_step(lattice, temp, nmax);
        energy[iter] = all_energy(lattice, nmax);
        order[iter] = get_order(lattice, nmax);
    }

    auto final_time = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(final_time- initial_time).count();

    std::cout << program << ": Size: " << nmax << ", Steps: " << nsteps << ", T*: " << temp << ", Order: " << order[nsteps-1] << ", Thread Count: " << threadcount << ", Time: " << runtime*1e-9 << std::endl;
    savedat(lattice, nsteps, temp, runtime*1e-9, ratio, energy, order, nmax);
    return EXIT_SUCCESS;
}



