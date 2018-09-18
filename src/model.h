#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "utils.h"

//ʹ������ݶ��½���(SGD)

#pragma once

namespace fastlr {

    //��������
enum RegType {
    L0, //��ʹ������
    L1, //L1����
    L2, //L2����
};

class FastLR {
    private:
        //ѧϰ��
        double lr; 
        //������
        RegType reg_type;
        //epoch
        uint32_t epoch;
        //������ϵ��
        double eta;
        //����
        std::vector<double> w;
        //ƫ��
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

        //��������
        void load_data(const std::string& fea_path);

        //����ת��
        RegType trans_type(const std::string& type);

        //��ȡ����
        void read_features(const std::string& line);

        //shuffle
        void read_lines(const std::string& fea_path,
                        std::vector<std::string>& lines);

        //ѵ��
        void train();

        //����w���ݶ�
        double calc_gradient_w(RegType type,
                               uint32_t lable, 
                               double predict_value,
                               double one_f,
                               double one_w);

        double calc_gradient_b(uint32_t label,
                               double predict_value);
        
        //����Ԥ��ֵ
        double calc_predict_value(const std::vector<double>& one_feature);

        //����loss
        double calc_avg_loss();

        //����L1����
        double calc_L1_Reg();

        //sgd, ��ÿһ�н����ݶȸ���
        void train_line(uint32_t label,
                        const std::vector<double>& one_feature);
        //save model
        void save_model();
};

}
