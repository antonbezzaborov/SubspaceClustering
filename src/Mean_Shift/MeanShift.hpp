#pragma once
#include <vector>
#include <string>

class MeanShift {
private:
    double bandwidth;
    bool use_gaussian;
    
    std::vector<std::vector<double>> data;
    int n_samples;
    int n_features;
    std::vector<int> labels;

    double euclideanDist(const std::vector<double>& a, const std::vector<double>& b);

public:
    // Constructor
    MeanShift(double b, bool gaussian);
    void loadData(const std::string& filename);
    void fit();
    void saveData(const std::string& labels_file);
};
