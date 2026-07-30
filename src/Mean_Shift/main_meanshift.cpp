#include "MeanShift.hpp"
#include <iostream>

using namespace std;

int main() {
    double bandwidth = 15.0;
    bool use_gaussian = true;
    
    cout << "--- Starting Mean Shift ---" << endl;
    
    MeanShift model(bandwidth, use_gaussian);
    model.loadData("input.csv");
    model.fit();
    model.saveData("labels_meanshift.csv");
    
    return 0;
}
