#pragma once
#include <vector>
#include <string>

class StandardKMeans {
private:
    int K;
    int max_iterations;
    bool use_kmeans_pp; 
    double epsilon = 1e-9;

    std::vector<std::vector<double>> data;
    int n_samples;
    int n_features;

    std::vector<std::vector<double>> centroids;
    std::vector<int> labels;

    void initRandom();
    void initKMeansPlusPlus();
    void classify();
    bool updateCentroids(); 

public:
    // Constructor
    StandardKMeans(int k, int iter, bool use_pp);
    void loadData(const std::string& filename);
    void fit();
    void saveData(const std::string& labels_file);
};
