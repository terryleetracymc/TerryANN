#ifndef NETWORK_H
#define NETWORK_H
#include "inc_files.h"
using namespace std;
using namespace arma;
//@初始化一个三层神经网络
//@本程序基于UFLDL教程
//@作者：李腾
//@最近完成时间：2013年9月26日
//@
class network
{
public:
    //训练中最小的一次误差
    float min_error;
    //最小误差矩阵
    mat mw1,mw2,mw3;
    //最小误差偏执
    rowvec mo1,mo2,mo3;
    //学习率（learning rate），动量系数（momentum coefficient）
    float lr;
    //该网络的信息,分别记录输入/输出/三层细胞的维数
    int in_vec,out_vec,ln1,ln2;
    //初始化一个三层神经网络,
    network(int l1,int l2);
    //确定输入输出维数
    void setVec(int in,int out);
    //更新输出
    void updateOutput(mat in);
    //sigmoid函数
    mat sigmoid(mat x);
    //该神经网络输入输出
    mat input,output;
    //各层的激活值
    //a3为输出的激活值
    mat a1,a2,a3;
    //各层的偏置向量
    rowvec o1,o2,o3;
    //各层细胞的残差
    mat e1,e2,e3;
    //隐藏层输出
    mat h1,h2;
    //各层细胞的权值矩阵
    mat w1,w2,w3;
    //权值变化矩阵
    mat dw1,dw2,dw3;
    //动量训练算法用到的参数
    //权值矩阵w(k)
    mat wk1,wk2,wk3;
    //权值矩阵w(k-1)
    mat wp1,wp2,wp3;
    //一次训练后的前后输出误差
    mat error1,error2;
    //导数
    mat delta1,delta2,delta3;
    //带动量的神经网络训练，如果单样本在线训练，单纯的BP神经网络训练
    void withMomentumTrain(mat sample,mat result);
    //样本训练，如果是多样本就批训练，如果单样本在线训练，单纯的BP神经网络训练
    void simplyBPTrain(mat sample,mat result);
    //带稀疏化参数的训练,rs1和rs2为隐藏层不受抑制细胞的个数，默认为0
    void withSparseTrain(mat sample,mat result,int rs1=0,int rs2=0);
    //存储当前的矩阵
    void save(string pre);
    network(string pre);
    //察看输出
    void active(rowvec input);
    //针对一个数据集集中训练
    //dataset训练集矩阵
    //训练集的每一行为一个数据样本，行的维数为输入维数+输出维数
    //该训练方法为在线训练，即出现一个训练样本就马上更新一次权重
    //TRAINNING_TYPE为训练的方法
    //MAX_TRAINNING_TIME是最多训练次数
    //后两个系数rs1和rs2是为稀疏训练提供的参数接口，其他训练方法可以忽略
    void datasetOnLineTrain(mat dataset,int TRAINNING_TYPE,int MAX_TRAINNING_TIME,int rs1=0,int rs2=0);
    //针对一个数据集集中训练
    //dataset训练集矩阵
    //训练集的每一行为一个数据样本，行的维数为输入维数+输出维数
    //该训练方法为批训练，即等所有样本出现以后更新一次权重
    //TRAINNING_TYPE为训练的方法
    //MAX_TRAINNING_TIME是最多训练次数
    //bat_size批大小
    void datasetBatTrain(mat dataset,int TRAINNING_TYPE,int MAX_TRAINNING_TIME,int bat_size=1,int rs1=0,int rs2=0);
protected:
private:
};

#endif // NETWORK_H
