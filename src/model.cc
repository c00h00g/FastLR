#include "model.h"

namespace fastlr {

FastLR::FastLR(double lr,
               const std::string& reg_type,
               uint32_t epoch,
               double eta,
               const std::string& fea_path) {
}

/**
 * @brief : 加载训练数据
 * 数据格式 : lable(0|1)\tf1\tf2\t...
 **/
void FastLR::load_data(const std::string& fea_path) {
    ifstream fin(fea_path.c_str());
    std::string line;

    while (getline(fin, line)) {
        read_features(line);
    }
}

/**
 * @brief : 读取特征的值
 **/
void FastLR::read_features(const std::vector<std::string>& line) {
    std::vector<std::string> f_info;
    split(line, f_info, is_any_of("\t"));
    assert(f_info.size() > 0);

    std::vector<double> one_feature;
    for (uint32_t i = 0; i < f_info.size(); ++i) {
        if (i == 0) {
            labels.push_back(std::stoi(f_info[i]));
        } else {
            one_feature.push_back(std::stod(f_info[i]));
        }
    }
    features.push_back(one_feature);
}

/**
 * @brief : 训练数据
 **/
void FastLR::train() {
    uint32_t f_num = labels.size();
    for (uint32_t i = 0; i < epoch; ++i) {
        for (uint32_t j = 0; j < f_num; ++j) {
            train_line(labels[i], features[i]);
        }
    }
}

/**
 * @brief : 随机梯度下降(SGD)
 **/
void FastLR::train_line(uint32_t label,
                        const std::vector<double>& one_feature) {
    uint32_t w_num = one_feature.size();
    for (uint32_t i = 0; i < w_num; ++i) {
        double predict = calc_predict_value(one_feature);
        double grad = calc_gradient(label, 
                                    predict, 
                                    one_feature[i], 
                                    w[i]);
        w[i] -= lr * grad;
    }
}

/**
 * @brief : 计算预测的值
 * @param one_feature : 所有的样本值
 **/
double FastLR::calc_predict_value(const std::vector<double>& one_feature) {
    double sum = 0.0;
    for (uint32_t i = 0; i < one_feature.size(); ++i) {
        sum += one_feature[i] * w[i];
    }
    sum += b;

    return 1.0 / (1 + std::exp(-1.0 * sum));
}

/**
 * @brief : 计算梯度
 * @param label
 * @param predict_value
 * @param one_f : 一个纬度上feature的值
 * @param one_w : 要梯度下降的参数
 **/
double FastLR::calc_gradient(RegType reg_type,
                             uint32_t lable, 
                             double predict_value,
                             double one_f,
                             double one_w) {
    double grad = 0.0;
    grad = (-1.0 * (lable * 1.0 - predict_value) * one_f + eta * one_w);
    return grad;
}

/**
 * @brief : 计算l2正则
 **/
double FastLR::calc_l2_regular() {

}

/**
 * @brief : 计算l1正则
 **/
double FastLR::calc_l1_regular() {
    
}


} //end namespace
