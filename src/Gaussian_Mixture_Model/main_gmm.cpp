#include "GMM.hpp"
#include <iostream>

using namespace std;

int main() {
    int K = 3;
    int iterations = 50;
    
    cout << "--- Starting Gaussian Mixture Model ---" << endl;
    
    GMM model(K, iterations);
    model.loadData("input.csv");
    model.fit();
    model.saveData("labels_gmm.csv");
    
    return 0;
}
