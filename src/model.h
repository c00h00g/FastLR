#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <boost/algorithm/string.hpp>

//使用随机梯度下降法(SGD)

#pragma once

namespace fastlr {

    //正则类型
enum RegType {
    L0, //不使用正则
    L1, //L1正则
    L2, //L2正则
};

class FastLR {
    private:
        //学习率
        double lr; 
        //正则项
        RegType reg_type;
        //epoch
        uint32_t epoch;
        //正则项系数
        double eta;
        //参数
        std::vector<double> w;
        //偏置
        double b;
        //features
        std::vector<std::vector<double> > features;
        //lable
        std::vector<uint32_t> labels;
        //loss
        double loss;

    publict:
        FastLR();
        //对参数进行更新
        void update_param();

        //加载数据
        void load_data();

        //训练
        void train();
};

}
