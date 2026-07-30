#include "DBSCAN.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

DBSCAN::DBSCAN(double e, int m) : eps(e), minPts(m) {}

void DBSCAN::loadData(const string& filename) {
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
    labels.assign(n_samples, 0); // 0 indicates unvisited
}

double DBSCAN::euclideanDistance(const vector<double>& a, const vector<double>& b) {
    double dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) dist += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(dist);
}

vector<int> DBSCAN::findNeighbors(int index) {
    vector<int> neighbors;
    for (int i = 0; i < n_samples; ++i) {
        if (euclideanDistance(data[index], data[i]) <= eps) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

void DBSCAN::fit() {
    int cluster_id = 0;
    for (int i = 0; i < n_samples; ++i) {
        if (labels[i] != 0) continue; // Already visited

        vector<int> neighbors = findNeighbors(i);
        if (neighbors.size() < minPts) {
            labels[i] = -1; // Mark as noise
            continue;
        }

        cluster_id++;
        labels[i] = cluster_id;

        // Expand cluster
        vector<int> seed_set = neighbors;
        for (size_t j = 0; j < seed_set.size(); ++j) {
            int current_p = seed_set[j];
            
            if (labels[current_p] == -1) labels[current_p] = cluster_id; // Change noise to border point
            if (labels[current_p] != 0) continue; // Skip if already processed

            labels[current_p] = cluster_id;
            vector<int> current_neighbors = findNeighbors(current_p);
            
            if (current_neighbors.size() >= minPts) {
                seed_set.insert(seed_set.end(), current_neighbors.begin(), current_neighbors.end());
            }
        }
    }
    cout << "DBSCAN finished. Found " << cluster_id << " clusters." << endl;
}

void DBSCAN::saveData(const string& l_file) {
    ofstream f_labels(l_file);
    for (int l : labels) f_labels << l << "\n";
}
