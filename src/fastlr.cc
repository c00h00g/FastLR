#include "model.h"
#include <string>
#include <iostream>
#include <memory>

int main() {
    std::shared_ptr<fastlr::FastLR> fast_lr =
           std::make_shared<fastlr::FastLR>(0.01, "L2", 30, 0.01, "./train.tst");
    std::cout << "Hello, FastLR!" << std::endl;
    fast_lr->train();
    return 0;
}
