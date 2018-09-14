#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <boost/algorithm/string.hpp>

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

    publict:
        FastLR();
        //�Բ������и���
        void update_param();

        //��������
        void load_data();

        //ѵ��
        void train();
};

}
