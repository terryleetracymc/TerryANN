#include "inc_files.h"
#include "network.h"
using namespace std;
using namespace arma;
int main()
{
    srand((unsigned int)time(NULL));
//    int layer_info[4]={103,10,10,9};
//    network nets(layer_info);
//    nets.initNet();
//    nets.save("before/cell");
//    network nets("before/cellm");
//    nets.lr=0.0001;
//    nets.tor_error=0.03;
//    nets.rnd_scale=0.0001;
//    nets.roup=0.015;
//    nets.sparse_para=0.8;
//    mat dataset;
//    dataset.load("sub_dataset.dat",raw_ascii);
//    nets.datasetOnLineTrain(dataset,BP_NONE,1000);
//    nets.save("after/cell");
    network nets("after/cellm");
    nets.active(rowvec(""));
    return 0;
}
