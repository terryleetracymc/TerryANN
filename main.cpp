#include <iostream>
#include <fstream>
#include <armadillo>
#include "network.h"
#include "armadillo_bits/config.hpp"
using namespace std;
using namespace arma;
int main()
{
    srand((unsigned int)time(NULL));
//    network nets(10,10);
//    nets.setVec(103,9);
    network nets("after/cell");
    mat dataset;
    dataset.load("deal_dataset.dat",raw_ascii);
    nets.datasetBatTrain(dataset,BP_WITH_MOM,3000,500);
    nets.save("before/cell");
    return 0;
}
