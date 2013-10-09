#include "inc_files.h"
#include "network.h"
using namespace std;
using namespace arma;
int main()
{
    srand((unsigned int)time(NULL));
    int li[4]={3,3,5,3};
    network net(li);
    net.initNet();
    net.lr=0.00009;
    net.rnd_scale=0.00005;
    net.sparse_para=0.5;
    net.roup=0.015;
    net.rs[2]=2;
    net.rs[1]=3;
    for(int i=0;i<1000;i++)
        net.withSparseTrain(rowvec("0 0.5 0.2"),rowvec("0.5 0.5 0.5"));
//    net.active(rowvec("0.9 0.5 0.2"));
//    net.active(rowvec("0 0 0"));
    net.active(rowvec("0 0.5 0.2"));
    cout<<net.out[1]<<net.out[2]<<net.out[3];
    net.save("before/cell");
//    net.active(rowvec("0.9 0 0.2"));
    return 0;
}
