#include "GMM.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <chrono>

using namespace std;

const double PI = 3.14159265358979323846;

GMM::GMM(int k, int iter) : K(k), max_iter(iter) {}

void GMM::loadData(const string& filename) {
    ifstream file(filename);
    string line, val;
    while (getline(file, line)) {
        vector<double> row;
        stringstream s(line);
        while (getline(s, val, ',')) row.push_back(stod(val));
        if (!row.empty()) data.push_back(row);
    }
    n_samples = data.size();
    n_features = data[0].size();
    labels.assign(n_samples, 0);
    p.assign(n_samples, vector<double>(K, 0.0));
}

// Gauss-Jordan matrix inversion
bool GMM::invertMatrix(vector<vector<double>> A, vector<vector<double>>& invOut, double& det) {
    int n = A.size();
    invOut.assign(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) invOut[i][i] = 1.0;
    det = 1.0;

    for (int i = 0; i < n; ++i) {
        double pivot = A[i][i];
        if (abs(pivot) < 1e-12) return false;
        det *= pivot;

        for (int j = 0; j < n; ++j) {
            A[i][j] /= pivot;
            invOut[i][j] /= pivot;
        }
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = A[k][i];
                for (int j = 0; j < n; ++j) {
                    A[k][j] -= factor * A[i][j];
                    invOut[k][j] -= factor * invOut[i][j];
                }
            }
        }
    }
    return true;
}

void GMM::calculatePosteriori() {
    for (int i = 0; i < n_samples; ++i) {
        double sum_p = 0.0;
        for (int j = 0; j < K; ++j) {
            // Add 1e-6 to diagonal for numerical stability (same as MATLAB code)
            vector<vector<double>> cov = sigma[j];
            for (int f = 0; f < n_features; ++f) cov[f][f] += 1e-6;

            vector<vector<double>> inv_cov;
            double det;
            if (!invertMatrix(cov, inv_cov, det)) det = 1e-6;

            double mahalanobis = 0.0;
            vector<double> diff(n_features);
            for (int f = 0; f < n_features; ++f) diff[f] = data[i][f] - miu[j][f];

            for (int r = 0; r < n_features; ++r) {
                for (int c = 0; c < n_features; ++c) {
                    mahalanobis += diff[r] * inv_cov[r][c] * diff[c];
                }
            }

            double coef = 1.0 / sqrt(pow(2 * PI, n_features) * abs(det));
            p[i][j] = alpha[j] * coef * exp(-0.5 * mahalanobis);
            sum_p += p[i][j];
        }
        if (sum_p < 1e-15) sum_p = 1e-15;
        for (int j = 0; j < K; ++j) p[i][j] /= sum_p;
    }
}

void GMM::fit() {
    unsigned seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, n_samples - 1);

    alpha.assign(K, 1.0 / K);
    miu.assign(K, vector<double>(n_features, 0.0));
    sigma.assign(K, vector<vector<double>>(n_features, vector<double>(n_features, 0.0)));

    for (int k = 0; k < K; ++k) {
        miu[k] = data[dis(gen)];
        for (int f = 0; f < n_features; ++f) sigma[k][f][f] = 1.0; // Identity matrix initialization
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        calculatePosteriori(); // E-Step

        // M-Step
        for (int j = 0; j < K; ++j) {
            double sum_p = 0.0;
            for (int i = 0; i < n_samples; ++i) sum_p += p[i][j];

            vector<double> t_miu(n_features, 0.0);
            for (int i = 0; i < n_samples; ++i) {
                for (int f = 0; f < n_features; ++f) t_miu[f] += p[i][j] * data[i][f];
            }
            for (int f = 0; f < n_features; ++f) miu[j][f] = t_miu[f] / sum_p;

            vector<vector<double>> t_sigma(n_features, vector<double>(n_features, 0.0));
            for (int i = 0; i < n_samples; ++i) {
                for (int r = 0; r < n_features; ++r) {
                    for (int c = 0; c < n_features; ++c) {
                        t_sigma[r][c] += p[i][j] * (data[i][r] - miu[j][r]) * (data[i][c] - miu[j][c]);
                    }
                }
            }
            for (int r = 0; r < n_features; ++r) {
                for (int c = 0; c < n_features; ++c) sigma[j][r][c] = t_sigma[r][c] / sum_p;
            }

            alpha[j] = sum_p / n_samples;
        }
    }

    // Final Classification (Maximum Likelihood)
    for (int i = 0; i < n_samples; ++i) {
        double max_val = -1.0;
        for (int j = 0; j < K; ++j) {
            if (p[i][j] > max_val) {
                max_val = p[i][j];
                labels[i] = j;
            }
        }
    }
    cout << "GMM finished." << endl;
}

void GMM::saveData(const string& l_file) {
    ofstream f_labels(l_file);
    for (int l : labels) f_labels << l << "\n";
}
