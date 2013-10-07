#include "network.h"

void network::withSparseTrain(mat sample,mat result,int rs1,int rs2)
{
    updateOutput(sample);
    error1=result-output;
    //隐藏层惩罚因子向量
    rowvec r1,r2;
    //经过处理后的惩罚因子向量
    rowvec ar1,ar2;
    //求出平均激活值
    r2=sum(h2)/sample.n_rows;
    r1=sum(h1)/sample.n_rows;

    ar2=SPARSE_PARA*(-ROUP/r2+(1-ROUP)/(1-r2));
    ar1=SPARSE_PARA*(-ROUP/r1+(1-ROUP)/(1-r1));

    if(rs1!=0)
        ar1.cols(0,rs1-1).fill(0);
    if(rs2!=0)
        ar2.cols(0,rs2-1).fill(0);

    //计算残差
    e3=-(result-output)%output%(1-output);
    //隐藏层加入稀疏性
    e2=e3*w3.t();
    e2.each_row()+=ar2;//SPARSE_PARA*(-ROUP/r2+(1-ROUP)/(1-r2));
    e2=e2%h2%(1-h2);

    e1=e2*w2.t();
    e1.each_row()+=ar1;//SPARSE_PARA*(-ROUP/r1+(1-ROUP)/(1-r1));
    e1=e1%h1%(1-h1);

    //批梯度
    delta1=input.t()*e1/sample.n_rows;
    delta2=h1.t()*e2/sample.n_rows;
    delta3=h2.t()*e3/sample.n_rows;

    //批梯度下降法+权值衰减
    w1=w1-lr*(delta1+LAMUDA*w1);
    w2=w2-lr*(delta2+LAMUDA*w2);
    w3=w3-lr*(delta3+LAMUDA*w3);

    //加入随机扰动
    mat rdmat1,rdmat2,rdmat3;
    rdmat1=w1;
    rdmat1.randu();
    rdmat1=(2*rdmat1-1)*RND_SCALAR_RADIO;
    w1=w1+rdmat1;

    rdmat2=w2;
    rdmat2.randu();
    rdmat2=(2*rdmat2-1)*RND_SCALAR_RADIO;
    w2=w2+rdmat2;

    rdmat3=w3;
    rdmat3.randu();
    rdmat3=(2*rdmat3-1)*RND_SCALAR_RADIO;
    w3=w3+rdmat3;

    //偏置修改
    o1=o1-sum(e1)/sample.n_rows;
    o2=o2-sum(e2)/sample.n_rows;
    o3=o3-sum(e3)/sample.n_rows;
}

network::network(string pre)
{
    string savePath=pre+"w1.dat";
    w1.load(savePath.c_str(),raw_ascii);
    savePath=pre+"o1.dat";
    o1.load(savePath.c_str(),raw_ascii);
    savePath=pre+"w2.dat";
    w2.load(savePath.c_str(),raw_ascii);
    savePath=pre+"o2.dat";
    o2.load(savePath.c_str(),raw_ascii);
    savePath=pre+"w3.dat";
    w3.load(savePath.c_str(),raw_ascii);
    savePath=pre+"o3.dat";
    o3.load(savePath.c_str(),raw_ascii);

    in_vec=w1.n_rows;
    ln1=w1.n_rows;
    ln2=w3.n_cols;
    out_vec=w3.n_rows;

    wk1=w1;
    wp1=w1;
    dw1=mat(in_vec,ln1);
    dw1.fill(0);

    wk3=w3;
    wp3=w3;
    dw3=mat(ln2,out_vec);
    dw3.fill(0);

    wk2=w2;
    wp2=w2;
    dw2=mat(ln1,ln2);
    dw2.fill(0);

    lr=LEARNING_RATE;
}

network::network(int l1,int l2)
{
    ln1=l1;
    ln2=l2;

    w2=mat(ln1,ln2);
    w2.randu();
    w2=1-2*w2;
    w2=w2*scalarRadio;

    wk2=w2;
    wp2=w2;
    dw2=mat(ln1,ln2);
    dw2.fill(0);

    o1=rowvec(1,ln1);
    o1.randu();
    o1=1-2*o1;
    o1=o1*scalarRadio;

    o2=rowvec(1,ln2);
    o2.randu();
    o2=1-2*o2;
    o2=o2*scalarRadio;

    lr=LEARNING_RATE;
}

void network::setVec(int in,int out)
{
    in_vec=in;
    out_vec=out;

    w1=mat(in_vec,ln1);
    w3=mat(ln2,out_vec);

    wk1=w1;
    wp1=w1;
    dw1=mat(in_vec,ln1);
    dw1.fill(0);

    wk3=w3;
    wp3=w3;
    dw3=mat(ln2,out_vec);
    dw3.fill(0);

    w1.randu();
    w3.randu();

    w1=1-2*w1;
    w1=w1*scalarRadio;

    w2=1-2*w2;
    w2=w2*scalarRadio;

    o3=rowvec(1,out_vec);
    o3.randu();
    o3=1-2*o3;
    o3=o3*scalarRadio;
}

mat network::sigmoid(mat x)
{
    return 1/(exp(-x)+1);
}

//矩阵形式样本更新
void network::updateOutput(mat in)
{
    input=in;
    a1=input*w1;
    a1.each_row()+=o1;
    h1=sigmoid(a1);

    a2=h1*w2;
    a2.each_row()+=o2;
    h2=sigmoid(a2);

    a3=h2*w3;
    a3.each_row()+=o3;
    output=sigmoid(a3);
}

void network::active(rowvec in)
{
    updateOutput(in);
    cout<<output;
}

//存储权值
void network::save(string pre)
{
    string savePath=pre+"w1.dat";
    w1.save(savePath.c_str(),SAVE_FORM);
    savePath=pre+"o1.dat";
    o1.save(savePath.c_str(),SAVE_FORM);
    savePath=pre+"w2.dat";
    w2.save(savePath.c_str(),SAVE_FORM);
    savePath=pre+"o2.dat";
    o2.save(savePath.c_str(),SAVE_FORM);
    savePath=pre+"w3.dat";
    w3.save(savePath.c_str(),SAVE_FORM);
    savePath=pre+"o3.dat";
    o3.save(savePath.c_str(),SAVE_FORM);
}

void network::simplyBPTrain(mat sample,mat result)
{
    //先使用样本前向更新
    updateOutput(sample);
    error1=result-output;
    //反馈更新权值
    //计算残差
    e3=-(result-output)%output%(1-output);
    e2=(e3*w3.t())%h2%(1-h2);
    e1=(e2*w2.t())%h1%(1-h1);

    //批梯度
    delta1=input.t()*e1/sample.n_rows;
    delta2=h1.t()*e2/sample.n_rows;
    delta3=h2.t()*e3/sample.n_rows;

    //批梯度下降法+权值衰减
    w1=w1-lr*(delta1+LAMUDA*w1);
    w2=w2-lr*(delta2+LAMUDA*w2);
    w3=w3-lr*(delta3+LAMUDA*w3);

    //加入随机扰动
    mat rdmat1,rdmat2,rdmat3;
    rdmat1=w1;
    rdmat1.randu();
    rdmat1=(2*rdmat1-1)*RND_SCALAR_RADIO;
    w1=w1+rdmat1;

    rdmat2=w2;
    rdmat2.randu();
    rdmat2=(2*rdmat2-1)*RND_SCALAR_RADIO;
    w2=w2+rdmat2;

    rdmat3=w3;
    rdmat3.randu();
    rdmat3=(2*rdmat3-1)*RND_SCALAR_RADIO;
    w3=w3+rdmat3;

    //偏置修改
    o1=o1-sum(e1)/sample.n_rows;
    o2=o2-sum(e2)/sample.n_rows;
    o3=o3-sum(e3)/sample.n_rows;
}

void network::withMomentumTrain(mat sample,mat result)
{
    updateOutput(sample);
    error1=result-output;
    //计算残差
    e3=-(result-output)%output%(1-output);
    e2=(e3*w3.t())%h2%(1-h2);
    e1=(e2*w2.t())%h1%(1-h1);

    //批梯度
    delta1=input.t()*e1/sample.n_rows;
    delta2=h1.t()*e2/sample.n_rows;
    delta3=h2.t()*e3/sample.n_rows;

    dw1=wk1-wp1;
    dw2=wk2-wp2;
    dw3=wk3-wp3;

    //批梯度下降法+权值衰减
    w1=w1-lr*(delta1+LAMUDA*w1)+ALAPA*dw1;
    w2=w2-lr*(delta2+LAMUDA*w2)+ALAPA*dw2;
    w3=w3-lr*(delta3+LAMUDA*w3)+ALAPA*dw3;

    //加入随机扰动
    mat rdmat1,rdmat2,rdmat3;
    rdmat1=w1;
    rdmat1.randu();
    rdmat1=(2*rdmat1-1)*RND_SCALAR_RADIO;
    w1=w1+rdmat1;

    rdmat2=w2;
    rdmat2.randu();
    rdmat2=(2*rdmat2-1)*RND_SCALAR_RADIO;
    w2=w2+rdmat2;

    rdmat3=w3;
    rdmat3.randu();
    rdmat3=(2*rdmat3-1)*RND_SCALAR_RADIO;
    w3=w3+rdmat3;

    //偏置修改
    o1=o1-sum(e1)/sample.n_rows;
    o2=o2-sum(e2)/sample.n_rows;
    o3=o3-sum(e3)/sample.n_rows;

    wp1=wk1;
    wp2=wk2;
    wp3=wk3;

    wk1=w1;
    wk2=w2;
    wk3=w3;
}

void network::datasetBatTrain(mat dataset,int TRAINNING_TYPE,int MAX_TRAINNING_TIME,int bat_size,int rs1,int rs2)
{
    int time=0;
    //上一轮训练下来的误差
    double round_error;
    //样本的输入和输出
    mat sample,out;
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
            else if(TRAINNING_TYPE==BP_WITH_MOM){
                withMomentumTrain(sample,out);
            }
            else if(TRAINNING_TYPE==BP_WITH_SPARSE){
                withSparseTrain(sample,out,rs1,rs2);
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
            else if(TRAINNING_TYPE==BP_WITH_MOM){
                withMomentumTrain(sample,out);
            }
            else if(TRAINNING_TYPE==BP_WITH_SPARSE){
                withSparseTrain(sample,out,rs1,rs2);
            }
        }
        updateOutput(dataset.cols(0,in_vec-1));
        error2=dataset.cols(in_vec,dataset.n_cols-1)-output;
        round_error=norm(sum(error2%error2/dataset.n_rows),2);
        time++;
        cout<<"第"<<time<<"次训练的误差变化为"<<setprecision(50)<<round_error<<endl;
        if(round_error<TOR_ERROR)
            break;
    }
}

void network::datasetOnLineTrain(mat dataset,int TRAINNING_TYPE,int MAX_TRAINNING_TIME,int rs1,int rs2)
{
    int time=0;
    //上一轮训练下来的误差
    double round_error;
    //样本的输入和输出
    rowvec sample,out;
    while(time<MAX_TRAINNING_TIME)
    {
        round_error=0;
        for(int i=0; i<dataset.n_rows; i++)
        {
            sample=dataset.row(i).cols(0,in_vec-1);
            out=dataset.row(i).cols(in_vec,dataset.n_cols-1);
            if(TRAINNING_TYPE==BP_NONE)
            {
                simplyBPTrain(sample,out);
            }
            else if(TRAINNING_TYPE==BP_WITH_MOM)
            {
                withMomentumTrain(sample,out);
            }
            else if(TRAINNING_TYPE==BP_WITH_SPARSE)
            {
                withSparseTrain(sample,out);
            }
        }
        updateOutput(dataset.cols(0,in_vec-1));
        error2=dataset.cols(in_vec,dataset.n_cols-1)-output;
        round_error=norm(sum(error2%error2/dataset.n_rows),2);
        time++;
        cout<<"第"<<time<<"次训练的误差变化为"<<setprecision(50)<<round_error<<endl;
        if(round_error<TOR_ERROR)
            break;
    }
}
