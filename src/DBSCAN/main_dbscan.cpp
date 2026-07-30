#include "DBSCAN.hpp"
#include <iostream>

using namespace std;

int main() {
    double eps = 15.0;
    int minPts = 5;
    
    cout << "--- Starting DBSCAN ---" << endl;
    
    DBSCAN model(eps, minPts);
    model.loadData("input.csv");
    model.fit();
    model.saveData("labels_dbscan.csv");
    
    return 0;
}
