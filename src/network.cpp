#include "network.h"

network::network(int layer_info[])
{
    layer_num=0;
    in_vec=layer_info[0];
    ln[0]=layer_info[0];
    for(int i=1; i<MAX_LAYER; i++)
    {
        //设置为0的话，结束
        if(layer_info[i]==0)
            break;
        layer_num++;
        ln[i]=layer_info[i];
    }
    out_vec=ln[layer_num];
}

void network::initNet(double scale)
{
    //自底向上初始化
    for(int i=1; i<=layer_num; i++)
    {
        //权值初始化
        w[i]=mat(ln[i-1],ln[i]);
        w[i].randu();
        w[i]=1-2*w[i];
        w[i]=scale*w[i];
        //偏置初始化
        o[i]=rowvec(ln[i]);
        o[i].randu();
        o[i]=1-2*o[i];
        o[i]=scale*o[i];
    }
}

void network::updateOut(mat sample)
{
    //自底向上更新
    input=sample;
    out[0]=sample;
    for(int i=1; i<=layer_num; i++)
    {
        a[i]=out[i-1]*w[i];
        a[i].each_row()+=o[i];
        out[i]=sigmoid(a[i]);
    }
    output=out[layer_num];
}

void network::active(rowvec input)
{
    updateOut(input);
    cout<<output;
}

mat network::sigmoid(mat x)
{
    return 1/(exp(-x)+1);
}

void network::save(string pre)
{
    char tmp[10];
    string savePath;
    for(int i=1; i<=layer_num; i++)
    {
        //存储权值矩阵和偏置向量
        sprintf(tmp,"%d",i);
        savePath=pre+"w"+tmp+".dat";
        w[i].save(savePath.c_str(),SAVE_FORM);
        savePath=pre+"o"+tmp+".dat";
        o[i].save(savePath.c_str(),SAVE_FORM);

        savePath=pre+"mw"+tmp+".dat";
        mw[i].save(savePath.c_str(),SAVE_FORM);
        savePath=pre+"mo"+tmp+".dat";
        mo[i].save(savePath.c_str(),SAVE_FORM);
    }
    rowvec netinfo;
    netinfo<<layer_num<<in_vec<<out_vec;
    savePath=pre+"netinfo.dat";
    netinfo.save(savePath.c_str(),SAVE_FORM);
    savePath=pre+"mnetinfo.dat";
    netinfo.save(savePath.c_str(),SAVE_FORM);
}

network::network(string pre) {
    char tmp[10];
    string savePath;
    rowvec netinfo;
    savePath=pre+"netinfo.dat";
    netinfo.load(savePath.c_str(),SAVE_FORM);
    layer_num=netinfo(0);
    in_vec=netinfo(1);
    out_vec=netinfo(2);
    for(int i=1; i<=layer_num; i++)
    {
        //存储权值矩阵和偏置向量
        sprintf(tmp,"%d",i);
        savePath=pre+"w"+tmp+".dat";
        w[i].load(savePath.c_str(),SAVE_FORM);
        savePath=pre+"o"+tmp+".dat";
        o[i].load(savePath.c_str(),SAVE_FORM);
    }
}

void network::simplyBPTrain(mat sample,mat result)
{
    //前向更新
    updateOut(sample);
    //计算顶层残差
    e[layer_num]=-(result-output)%output%(1-output);
    for(int i=layer_num-1; i>=1; i--)
    {
        //计算隐藏层残差
        e[i]=(e[i+1]*w[i+1].t())%out[i]%(1-out[i]);
    }
    //计算梯度并修改权值
    mat rdmat[MAX_LAYER];
    for(int i=layer_num; i>=1; i--)
    {
        delta[i]=out[i-1].t()*e[i]/sample.n_rows;
        w[i]=w[i]-lr*delta[i];
        //随机扰动
        rdmat[i]=w[i];
        rdmat[i].randu();
        rdmat[i]=(2*rdmat[i]-1)*rnd_scale;
        w[i]=w[i]+rdmat[i];
        //偏置修改
        o[i]=o[i]-sum(e[i])/sample.n_rows;
    }
}

void network::withSparseTrain(mat sample,mat result)
{
    updateOut(sample);
    //隐藏层惩罚因子向量
    rowvec r[4];
    //经过处理后的惩罚因子向量
    rowvec ar[4];
    if(1==layer_num)
    {
        //没有隐藏层，进行BP训练
        simplyBPTrain(sample,result);
    }
    else
    {
        //对隐藏层处理
        for(int i=layer_num-1; i>=1; i--)
        {
            //求出平均激活值
            r[i]=sum(out[i])/sample.n_rows;
            ar[i]=sparse_para*(-roup/r[i]+(1-roup)/(1-r[i]));
            if(rs[i]!=0)
                ar[i].cols(0,rs[i]-1).fill(0);
        }
        //计算顶层残差
        e[layer_num]=-(result-output)%output%(1-output);
        for(int i=layer_num-1;i>=1;i--){
            e[i]=e[i+1]*w[i+1].t();
            e[i].each_row()+=ar[i];
            e[i]=e[i]%out[i]%(1-out[i]);
        }
        //计算梯度并修改权值
        mat rdmat[MAX_LAYER];
        for(int i=layer_num;i>=1;i--){
            delta[i]=out[i-1].t()*e[i]/sample.n_rows;
            w[i]=w[i]-lr*delta[i];
            //随机扰动
            rdmat[i]=w[i];
            rdmat[i].randu();
            rdmat[i]=(2*rdmat[i]-1)*rnd_scale;
            w[i]=w[i]+rdmat[i];
            //偏置修改
            o[i]=o[i]-sum(e[i])/sample.n_rows;
        }
    }
}

void network::datasetOnLineTrain(mat dataset,int TRAINNING_TYPE,int MAX_TRAINNING_TIME) {
    int time=0;
    //上一轮训练下来的误差
    double round_error;
    mat error2;
    //赋值训练最小误差为较大的数
    min_error=255;
    //样本的输入和输出
    rowvec sample,out;
    while(time<MAX_TRAINNING_TIME)
    {
        round_error=0;
        for(int i=0;i<dataset.n_rows;i++)
        {
            sample=dataset.row(i).cols(0,in_vec-1);
            out=dataset.row(i).cols(in_vec,dataset.n_cols-1);
            if(TRAINNING_TYPE==BP_NONE)
            {
                simplyBPTrain(sample,out);
            }
            else if(TRAINNING_TYPE==BP_WITH_SPARSE)
            {
                withSparseTrain(sample,out);
            }
            else{
                return;
            }
        }
        updateOut(dataset.cols(0,in_vec-1));
        error2=dataset.cols(in_vec,dataset.n_cols-1)-output;
        round_error=norm(sum(error2%error2/dataset.n_rows),2);
        time++;
        if(round_error<min_error)
        {
            min_error=round_error;
            for(int i=layer_num;i>=1;i--)
            {
                mw[i]=w[i];
                mo[i]=o[i];
            }
        }
        if(time%10==0)
            cout<<"第"<<time<<"次训练的误差变化为"<<setprecision(50)<<round_error<<endl;
        if(round_error<tor_error)
            break;
    }
    cout<<"本次训练了"<<time<<"次，最小训练误差为"<<min_error<<endl<<"最终训练误差为"<<round_error<<endl;
}

void network::datasetBatTrain(mat dataset,int TRAINNING_TYPE,int MAX_TRAINNING_TIME,int bat_size) {
    int time=0;
    //上一轮训练下来的误差
    double round_error;
    mat error2;
    //样本的输入和输出
    mat sample,out;
    min_error=255;
    int chunks,remain,start,end;
    chunks=dataset.n_rows/bat_size;
    remain=dataset.n_rows%bat_size;
    while(time<MAX_TRAINNING_TIME)
    {
        for(int i=0;i<chunks;i++)
        {
            start=i*bat_size;
            end=(i+1)*bat_size-1;
            sample=dataset.rows(start,end).cols(0,in_vec-1);
            out=dataset.rows(start,end).cols(in_vec,dataset.n_cols-1);
            if(TRAINNING_TYPE==BP_NONE)
            {
                simplyBPTrain(sample,out);
            }
            else if(TRAINNING_TYPE==BP_WITH_SPARSE){
                withSparseTrain(sample,out);
            }
            else{
                return;
            }
        }
        //剩下的样本训练
        if(remain!=0){
            start=chunks*bat_size;
            end=dataset.n_rows-1;
            sample=dataset.rows(start,end).cols(0,in_vec-1);
            out=dataset.rows(start,end).cols(in_vec,dataset.n_cols-1);
            if(TRAINNING_TYPE==BP_NONE)
            {
                simplyBPTrain(sample,out);
            }
            else if(TRAINNING_TYPE==BP_WITH_SPARSE){
                withSparseTrain(sample,out);
            }
            else{
                return;
            }
        }
        updateOut(dataset.cols(0,in_vec-1));
        error2=dataset.cols(in_vec,dataset.n_cols-1)-output;
        round_error=norm(sum(error2%error2/dataset.n_rows),2);
        time++;
        if(round_error<min_error)
        {
            min_error=round_error;
            for(int i=layer_num;i>=1;i--)
            {
                mw[i]=w[i];
                mo[i]=o[i];
            }
        }
        cout<<"第"<<time<<"次训练的误差变化为"<<setprecision(50)<<round_error<<endl;
        if(round_error<tor_error)
        {
            error=round_error;
            break;
        }
    }
}

network::~network()
{
}
