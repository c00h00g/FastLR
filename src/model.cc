#include "model.h"

namespace fastlr {

FastLR::FastLR(double lr,
               const std::string& reg_type,
               uint32_t epoch,
               double eta,
               const std::string& fea_path) {
    this->lr = lr;
    this->reg_type = trans_type(reg_type);
    this->epoch = epoch;
    this->eta = eta;
    b = 0;

    //加载数据
    load_data(fea_path);
}

/**
 * @brief : 类型转换
 * @param type
 **/
RegType FastLR::trans_type(const std::string& type) {
    if (type == "L1") {
        return L1;
    } else if (type == "L2") {
        return L2;
    } else {
        return L0;
    }
}

/**
 * @brief : 加载训练数据
 * 数据格式 : lable(0|1)\tf1\tf2\t...
 **/
void FastLR::load_data(const std::string& fea_path) {
    std::ifstream fin(fea_path.c_str());
    std::string line;

    while (getline(fin, line)) {
        read_features(line);
    }
}

/**
 * @brief : 读取特征的值
 **/
void FastLR::read_features(const std::string& line) {
    std::vector<std::string> f_info;
    split(line, f_info, "\t");
    assert(f_info.size() > 0);

    //对w分配空间
    if (w.size() == 0) {
        w.resize(f_info.size() - 1);
    }

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

/*
 * @brief : 训练数据
 **/
void FastLR::train() {
    //轮次
    while (epoch--) {
        uint32_t f_row = features.size();
        for (uint32_t i = 0; i < f_row; ++i) {
            train_line(labels[i], features[i]);
        }
    }
    for (uint32_t i = 0; i < w.size(); ++i) {
        std::cout << "w_" << i << ":" << w[i] << std::endl;
    }
    std::cout << "b:" << b << std::endl;
}

double FastLR::calc_avg_loss() {
    double loss = 0;
    uint32_t lable_num = labels.size();
    for (uint32_t i = 0; i < lable_num; ++i) {
        std::vector<double>& one_feature = features[i];
        double label = labels[i];
        double predict = calc_predict_value(one_feature);
        double one_loss = -1.0 * (label * std::log(predict) + (1 - label) * std::log(1 - predict));
        loss += one_loss;
        //std::cout << "predict is:" << predict << std::endl;
    }
    return loss * 1.0 / lable_num;
}

/**
 * @brief : 随机梯度下降(SGD)
 **/
void FastLR::train_line(uint32_t label,
                        const std::vector<double>& one_feature) {
    uint32_t w_num = one_feature.size();
    double predict = calc_predict_value(one_feature);

    //w梯度下降
    for (uint32_t i = 0; i < w_num; ++i) {
        double grad_w = calc_gradient_w(reg_type,
                                        label, 
                                        predict, 
                                        one_feature[i], 
                                        w[i]);
        w[i] -= lr * grad_w;
    }

    //b梯度下降
    double grad_b = calc_gradient_b(label, predict);
    b -= lr * grad_b;

    double loss = calc_avg_loss();
    std::cout << "loss is:" << loss << std::endl;

    return;
}

/**
 * @brief : 计算b的梯度
 **/
double FastLR::calc_gradient_b(uint32_t label, 
                               double predict_value) {
    double grad = -1.0 * (label - predict_value);
    return grad;
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
 * @brief : 计算L1正则
 **/
double FastLR::calc_L1_Reg() {
    double sum = 0;
    for (uint32_t i = 0; i < w.size(); ++i) {
        sum += w[i];
    }
    return sum > 0 ? 1 : -1;
}

/**
 * @brief : 计算梯度
 * @param label
 * @param predict_value
 * @param one_f : 一个纬度上feature的值
 * @param one_w : 要梯度下降的参数
 **/
double FastLR::calc_gradient_w(RegType type,
                               uint32_t lable, 
                               double predict_value,
                               double one_f,
                               double one_w) {
    double reg_value = 0.0;
    switch (type) {
        case L0:
            reg_value = 0.0;
            break;
        case L1:
            reg_value = eta * calc_L1_Reg();
            break;
        case L2:
            reg_value = eta * one_w;
            break;
        default:
            break;
    }

    double grad = 0.0;
    grad = (-1.0 * (lable * 1.0 - predict_value) * one_f + reg_value);
    return grad;
}

/**
 * @brief : 保存训练好的模型
 **/
void FastLR::save_model() {
    
}

} //end namespace
