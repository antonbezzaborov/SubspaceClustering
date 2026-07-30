#pragma once
#include <vector>
#include <string>

class DBSCAN {
private:
    double eps;
    int minPts;
    
    std::vector<std::vector<double>> data;
    int n_samples;
    int n_features;
    std::vector<int> labels;

    double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<int> findNeighbors(int index);

public:
    // Constructor
    DBSCAN(double e, int m);
    void loadData(const std::string& filename);
    void fit();
    void saveData(const std::string& labels_file);
};
