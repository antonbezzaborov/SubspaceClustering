#include "StandardKMeans.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <limits>
#include <chrono>

using namespace std;

StandardKMeans::StandardKMeans(int k, int iter, bool use_pp) 
    : K(k), max_iterations(iter), use_kmeans_pp(use_pp) {}

void StandardKMeans::loadData(const string& filename) {
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

void StandardKMeans::initRandom() {
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, n_samples - 1);

    centroids.assign(K, vector<double>(n_features, 0.0));
    for (int k = 0; k < K; ++k) centroids[k] = data[dis(gen)];
}

void StandardKMeans::initKMeansPlusPlus() {
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, n_samples - 1);
    
    centroids.clear();
    centroids.push_back(data[dis(gen)]); // First centroid is random

    for (int k = 1; k < K; ++k) {
        vector<double> min_sq_dists(n_samples, numeric_limits<double>::max());
        double total_dist = 0.0;

        for (int i = 0; i < n_samples; ++i) {
            for (const auto& c : centroids) {
                double dist = 0.0;
                for (int j = 0; j < n_features; ++j) {
                    dist += (data[i][j] - c[j]) * (data[i][j] - c[j]);
                }
                if (dist < min_sq_dists[i]) min_sq_dists[i] = dist;
            }
            total_dist += min_sq_dists[i];
        }

        // Roulette wheel selection
        uniform_real_distribution<> rand_prob(0.0, total_dist);
        double target = rand_prob(gen);
        double current_sum = 0.0;
        
        for (int i = 0; i < n_samples; ++i) {
            current_sum += min_sq_dists[i];
            if (current_sum >= target) {
                centroids.push_back(data[i]);
                break;
            }
        }
    }
}

void StandardKMeans::classify() {
    for (int i = 0; i < n_samples; ++i) {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = 0;

        for (int k = 0; k < K; ++k) {
            double dist = 0.0;
            for (int j = 0; j < n_features; ++j) {
                double diff = data[i][j] - centroids[k][j];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        labels[i] = best_cluster;
    }
}

bool StandardKMeans::updateCentroids() {
    vector<vector<double>> new_centroids(K, vector<double>(n_features, 0.0));
    vector<int> counts(K, 0);

    for (int i = 0; i < n_samples; ++i) {
        int k = labels[i];
        counts[k]++;
        for (int j = 0; j < n_features; ++j) new_centroids[k][j] += data[i][j];
    }

    bool unchanged = true;
    for (int k = 0; k < K; ++k) {
        if (counts[k] > 0) {
            for (int j = 0; j < n_features; ++j) {
                double new_val = new_centroids[k][j] / counts[k];
                if (abs(centroids[k][j] - new_val) > epsilon) unchanged = false;
                centroids[k][j] = new_val;
            }
        }
    }
    return unchanged;
}

void StandardKMeans::fit() {
    if (use_kmeans_pp) initKMeansPlusPlus();
    else initRandom();

    for (int iter = 0; iter < max_iterations; ++iter) {
        classify();
        if (updateCentroids()) {
            cout << "Converged after " << iter + 1 << " iterations.\n";
            break;
        }
    }
}

void StandardKMeans::saveData(const string& l_file) {
    ofstream f_labels(l_file);
    for (int l : labels) f_labels << l << "\n";
}
