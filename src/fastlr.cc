#include "model.h"
#include <string>
#include <iostream>
#include <memory>

int main() {
    std::shared_ptr<fastlr::FastLR> fast_lr =
           std::make_shared<fastlr::FastLR>(0.05, "L0", 1, 0.05, "./train.tst");
    std::cout << "Hello, FastLR!" << std::endl;
    return 0;
}
