#include "inc_files.h"
#include "network.h"
using namespace std;
using namespace arma;
int main()
{
//    srand((unsigned int)time(NULL));
//    int layer_info[4]={103,10,9,9};
////    network nets(layer_info);
////    nets.initNet();
////    nets.save("before/cell");
//    network nets("before/cellm");
//    nets.lr=0.9;
//    nets.tor_error=0.035;
//    nets.rnd_scale=0.0001;
//    nets.roup=0.015;
//    nets.sparse_para=0.8;
//    mat dataset;
//    dataset.load("sub_dataset.dat",raw_ascii);
//    nets.datasetBatTrain(dataset,BP_NONE,50000,100);
//    nets.save("after/cell");
//    network nets("after/cell");
//    nets.active(rowvec(""));
//    mat dataset;
//    rowvec maxV,minV;
//    dataset.load("house_8L.data",raw_ascii);
//    maxV=max(dataset);
//    minV=min(dataset);
//    dataset.each_row()/=(maxV-minV);
//    dataset.save("da.data",raw_ascii);
//    mat dataset;
//    dataset.load("trainset.data",raw_ascii);
//    int layer_info[4]={8,10,10,1};
//    network nets(layer_info);
//    nets.initNet();
//    nets.save("houseb/cell");
//    nets.lr=0.9;
//    nets.tor_error=0.001;
//    nets.rnd_scale=0.0001;
//    nets.roup=0.015;
//    nets.sparse_para=0.8;
//    nets.datasetBatTrain(dataset,BP_NONE,10000,100);
//    nets.save("housea/cell");
    network nets("housea/cell");
    nets.active(rowvec(""));
    return 0;
}
