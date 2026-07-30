#include "MeanShift.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

MeanShift::MeanShift(double b, bool gaussian) : bandwidth(b), use_gaussian(gaussian) {}

void MeanShift::loadData(const string& filename) {
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
}

double MeanShift::euclideanDist(const vector<double>& a, const vector<double>& b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += (a[i]-b[i])*(a[i]-b[i]);
    return sqrt(d);
}

void MeanShift::fit() {
    vector<vector<double>> destinations(n_samples, vector<double>(n_features, 0.0));
    
    for (int i = 0; i < n_samples; ++i) {
        vector<double> current = data[i];
        double shift_dist = 1.0;
        
        while (shift_dist > 1e-4) {
            vector<double> new_pos(n_features, 0.0);
            double total_weight = 0.0;

            for (int j = 0; j < n_samples; ++j) {
                double d = euclideanDist(current, data[j]);
                if (d <= bandwidth) {
                    double weight = 1.0;
                    if (use_gaussian) {
                        weight = exp(-(d * d) / (2 * bandwidth * bandwidth));
                    }
                    for (int f = 0; f < n_features; ++f) new_pos[f] += data[j][f] * weight;
                    total_weight += weight;
                }
            }

            if (total_weight > 0) {
                for (int f = 0; f < n_features; ++f) new_pos[f] /= total_weight;
            } else {
                new_pos = current; 
            }

            shift_dist = euclideanDist(current, new_pos);
            current = new_pos;
        }
        destinations[i] = current;
    }

    // Grouping nearby destination points into clusters
    vector<vector<double>> unique_centroids;
    for (int i = 0; i < n_samples; ++i) {
        int cluster_id = -1;
        for (size_t c = 0; c < unique_centroids.size(); ++c) {
            if (euclideanDist(destinations[i], unique_centroids[c]) < 1e-3) {
                cluster_id = c;
                break;
            }
        }
        if (cluster_id == -1) {
            unique_centroids.push_back(destinations[i]);
            cluster_id = unique_centroids.size() - 1;
        }
        labels[i] = cluster_id;
    }
    cout << "Mean Shift finished. Found " << unique_centroids.size() << " classes." << endl;
}

void MeanShift::saveData(const string& l_file) {
    ofstream f_labels(l_file);
    for (int l : labels) f_labels << l << "\n";
}
