#include "StandardKMeans.hpp"
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    string mode = (argc > 1) ? argv[1] : "kmeans++";
    
    int K = 3;
    int iterations = 100;
    bool use_pp = (mode == "kmeans++");

    cout << "--- Starting Standard K-Means ---" << endl;
    cout << "Mode: " << (use_pp ? "K-Means++" : "Random Init") << endl;
    
    StandardKMeans model(K, iterations, use_pp);
    model.loadData("input.csv");
    model.fit();
    model.saveData("labels_kmeans.csv");
    
    return 0;
}
