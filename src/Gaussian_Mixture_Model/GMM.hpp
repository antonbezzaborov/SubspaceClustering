#pragma once
#include <vector>
#include <string>

class GMM {
private:
    int K;
    int max_iter;
    
    std::vector<std::vector<double>> data;
    int n_samples;
    int n_features;
    std::vector<int> labels;

    std::vector<double> alpha; // Mixing coefficients
    std::vector<std::vector<double>> miu; // Means
    std::vector<std::vector<std::vector<double>>> sigma; // Covariance matrices
    std::vector<std::vector<double>> p; // Posteriori probabilities

    bool invertMatrix(std::vector<std::vector<double>> A, std::vector<std::vector<double>>& invOut, double& det);
    void calculatePosteriori();

public:
    // Constructor
    GMM(int k, int iter);
    void loadData(const std::string& filename);
    void fit();
    void saveData(const std::string& labels_file);
};
