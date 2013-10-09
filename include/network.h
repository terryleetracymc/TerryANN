#ifndef NETWORK_H
#define NETWORK_H
#include "inc_files.h"
using namespace std;
using namespace arma;
//@初始化一个单层，双层或者三层的神经网络
//@本程序基于UFLDL教程
//@作者：李腾
//@最近完成时间：2013年10月9日
//@
class network
{
public:
    //多次训练中最小误差
    double min_error;
    //学习率(learning rate)
    double lr;
    //一次训练后的输出误差,before_error,after_error
    double error;
    //稀疏项参数
    double sparse_para;
    //稀疏限制范围
    double roup;
    //隐藏层不被抑制的细胞个数(只针对稀疏训练)
    int rs[MAX_LAYER];
    //随机扰动尺度
    double rnd_scale;
    //容忍训练误差
    double tor_error;
    //层数
    int layer_num;
    //最小误差矩阵
    mat mw[MAX_LAYER];
    //最小误差偏置
    rowvec mo[MAX_LAYER];
    //各层的细胞数
    int ln[MAX_LAYER];
    //各层的激活值
    mat a[MAX_LAYER];
    //各层的偏置向量
    rowvec o[MAX_LAYER];
    //各层权值矩阵
    mat w[MAX_LAYER];
    //各层输出
    mat out[MAX_LAYER];
    //各层细胞残差
    mat e[MAX_LAYER];
    //导数
    mat delta[MAX_LAYER];
    //输入
    mat input,output;
    //输入输出维数
    int in_vec,out_vec;
    //用数组初始化神经网络
    //x为输入维数，最后一个不为零的数为输出维数
    //一层神经网络数组示例 layer_info[]={x,3,0,0}
    //二层示例 layer_info[]={x,4,3,0}
    //三层示例 layer_info[]={x,4,3,4}
    network(int layer_info[]);
    //初始化网络，利用尺度初始化网络权值参数
    void initNet(double scale=1);
    //更新输出
    void updateOut(mat sample);
    //存储权值
    void save(string pre);
    //读取网络
    network(string pre);
    //sigmoid函数,激活函数
    mat sigmoid(mat x);
    //激励
    void active(rowvec input);
    //*************************************
    //*******训练之前请自行设置好各种参数*******
    //*************************************
    //简单的BP网络训练
    void simplyBPTrain(mat sample,mat result);
    //带动量的BP网络训练,考虑要不要？**********************************
    //void withMomentumTrain(mat sample,mat result);
    //带稀疏参数的BP网络训练
    void withSparseTrain(mat sample,mat result);
    //针对一个数据集集中训练
    //dataset训练集矩阵
    //训练集的每一行为一个数据样本，行的维数为输入维数+输出维数
    //该训练方法为在线训练，即出现一个训练样本就马上更新一次权重
    //TRAINNING_TYPE为训练的方法
    //MAX_TRAINNING_TIME是最多训练次数
    void datasetOnLineTrain(mat dataset,int TRAINNING_TYPE,int MAX_TRAINNING_TIME);
    //针对一个数据集集中训练
    //dataset训练集矩阵
    //训练集的每一行为一个数据样本，行的维数为输入维数+输出维数
    //该训练方法为批训练，即等所有样本出现以后更新一次权重
    //TRAINNING_TYPE为训练的方法
    //MAX_TRAINNING_TIME是最多训练次数
    //bat_size批大小,当bat_size=1时变成在线训练……
    void datasetBatTrain(mat dataset,int TRAINNING_TYPE,int MAX_TRAINNING_TIME,int bat_size=1);
    virtual ~network();
protected:
private:
};

#endif // NETWORK_H
