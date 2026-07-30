#pragma once
#include <vector>
#include <string>

class SubspaceKMeans {
private:
    int K;
    int max_iterations;
    double beta;
    double lambda; // Parameter for the Entropy method
    bool use_entropy; // Method selection flag
    double epsilon = 1e-9;

    std::vector<std::vector<double>> data;
    int n_samples;
    int n_features;

    std::vector<std::vector<double>> centroids;
    std::vector<std::vector<double>> weights; 
    std::vector<int> labels;

    void initialize();
    void classify();
    void updateCentroids();
    double updateWeightsStandard();
    double updateWeightsEntropy();

public:
    // Constructor
    SubspaceKMeans(int k, int iter, double b, double l, bool entropy_mode);
    void loadData(const std::string& filename);
    void fit();
    void saveData(const std::string& labels_file, const std::string& weights_file);
};