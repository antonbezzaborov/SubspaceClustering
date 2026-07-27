#include "SubspaceKMeans.hpp"
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    string mode = (argc > 1) ? argv[1] : "standard";
    
    int K = 3;
    int iterations = 50;
    double beta = 2.0;
    double lambda = 100.0; // For entropy
    
    bool use_entropy = (mode == "entropy");

    cout << "--- Starting Subspace K-Means ---" << endl;
    cout << "Mode: " << (use_entropy ? "Entropy Weighting" : "Standard (Beta)") << endl;
    
    SubspaceKMeans model(K, iterations, beta, lambda, use_entropy);
    model.loadData("input.csv");
    model.fit();
    model.saveData("labels.csv", "weights.csv");
    
    return 0;
}