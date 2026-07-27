#include "SubspaceKMeans.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <limits>
#include <iomanip>
#include <chrono>

using namespace std;

SubspaceKMeans::SubspaceKMeans(int k, int iter, double b, double l, bool entropy_mode) 
    : K(k), max_iterations(iter), beta(b), lambda(l), use_entropy(entropy_mode) {}

void SubspaceKMeans::loadData(const string& filename) {
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
    labels.resize(n_samples, 0);
}

void SubspaceKMeans::initialize() {
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, n_samples - 1);

    centroids.assign(K, vector<double>(n_features, 0.0));
    for (int k = 0; k < K; ++k) centroids[k] = data[dis(gen)];

    double init_weight = 1.0 / n_features;
    weights.assign(K, vector<double>(n_features, init_weight));
}

void SubspaceKMeans::classify() {
    for (int i = 0; i < n_samples; ++i) {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = 0;

        for (int k = 0; k < K; ++k) {
            double dist = 0.0;
            for (int j = 0; j < n_features; ++j) {
                double diff = centroids[k][j] - data[i][j];
                if (use_entropy) {
                    dist += weights[k][j] * diff * diff; // Entropy weight
                } else {
                    dist += pow(weights[k][j], beta) * diff * diff; // Standard weight (beta)
                }
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        labels[i] = best_cluster;
    }
}

void SubspaceKMeans::updateCentroids() {
    vector<vector<double>> new_centroids(K, vector<double>(n_features, 0.0));
    vector<int> counts(K, 0);

    for (int i = 0; i < n_samples; ++i) {
        int k = labels[i];
        counts[k]++;
        for (int j = 0; j < n_features; ++j) new_centroids[k][j] += data[i][j];
    }

    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, n_samples - 1);


    for (int k = 0; k < K; ++k) {
        if (counts[k] > 0) {
            for (int j = 0; j < n_features; ++j) centroids[k][j] = new_centroids[k][j] / counts[k];
        } else {
            centroids[k] = data[dis(gen)];
        }
    }
}

// 1. Standard method
double SubspaceKMeans::updateWeightsStandard() {
    double J = 0.0;
    for (int k = 0; k < K; ++k) {
        vector<double> t_sum(n_features, epsilon);
        for (int i = 0; i < n_samples; ++i) {
            if (labels[i] == k) {
                for (int j = 0; j < n_features; ++j) {
                    double diff = centroids[k][j] - data[i][j];
                    t_sum[j] += diff * diff;
                }
            }
        }

        double power = 1.0 / (beta - 1.0);
        double sum_weights = 0.0;
        for (int j = 0; j < n_features; ++j) {
            J += pow(weights[k][j], beta) * t_sum[j];
            double t_denominator = 0.0;
            for (int p = 0; p < n_features; ++p) t_denominator += pow(t_sum[j] / t_sum[p], power);
            weights[k][j] = 1.0 / (t_denominator + epsilon);
            sum_weights += weights[k][j];
        }
        for (int j = 0; j < n_features; ++j) weights[k][j] /= sum_weights;
    }
    return J;
}

// 2. Entropy Method
double SubspaceKMeans::updateWeightsEntropy() {
    double J = 0.0;
    for (int k = 0; k < K; ++k) {
        vector<double> t_sum(n_features, 0.0);
        for (int i = 0; i < n_samples; ++i) {
            if (labels[i] == k) {
                for (int j = 0; j < n_features; ++j) {
                    double diff = centroids[k][j] - data[i][j];
                    t_sum[j] += diff * diff;
                }
            }
        }

        double sum_exp = 0.0;
        for (int j = 0; j < n_features; ++j) {
            weights[k][j] = exp(-t_sum[j] / lambda); // Entropy weight formula
            sum_exp += weights[k][j];
        }
        
        for (int j = 0; j < n_features; ++j) {
            weights[k][j] /= (sum_exp + epsilon); // Normalization
            J += weights[k][j] * t_sum[j] + lambda * weights[k][j] * log(weights[k][j] + epsilon);
        }
    }
    return J;
}

void SubspaceKMeans::fit() {
    initialize();
    double J_prev = -1.0;
    for (int iter = 0; iter < max_iterations; ++iter) {
        classify();
        updateCentroids();
        double J = use_entropy ? updateWeightsEntropy() : updateWeightsStandard();
        
        cout << "Iter " << iter + 1 << " | Loss (J): " << fixed << setprecision(4) << J << endl;
        if (abs(J - J_prev) < epsilon) {
            cout << "Converged after " << iter + 1 << " iterations.\n";
            break;
        }
        J_prev = J;
    }
}

void SubspaceKMeans::saveData(const string& l_file, const string& w_file) {
    ofstream f_labels(l_file);
    for (int l : labels) f_labels << l << "\n";

    ofstream f_weights(w_file);
    for (const auto& w_row : weights) {
        for (size_t i = 0; i < w_row.size(); ++i) {
            f_weights << w_row[i] << (i == w_row.size() - 1 ? "" : ",");
        }
        f_weights << "\n";
    }
}