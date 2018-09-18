#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "utils.h"

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

    public:
        FastLR(double lr,
               const std::string& reg_type,
               uint32_t epoch,
               double eta,
               const std::string& fea_path);

        //加载数据
        void load_data(const std::string& fea_path);

        //类型转换
        RegType trans_type(const std::string& type);

        //读取特征
        void read_features(const std::string& line);

        //shuffle
        void read_lines(const std::string& fea_path,
                        std::vector<std::string>& lines);

        //训练
        void train();

        //计算w的梯度
        double calc_gradient_w(RegType type,
                               uint32_t lable, 
                               double predict_value,
                               double one_f,
                               double one_w);

        double calc_gradient_b(uint32_t label,
                               double predict_value);
        
        //计算预测值
        double calc_predict_value(const std::vector<double>& one_feature);

        //计算loss
        double calc_avg_loss();

        //计算L1正则
        double calc_L1_Reg();

        //sgd, 对每一行进行梯度更新
        void train_line(uint32_t label,
                        const std::vector<double>& one_feature);
        //save model
        void save_model();
};

}
