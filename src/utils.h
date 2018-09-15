#include <string>

#pragma once

namespace fastlr {

void split(const std::string & input,
           std::vector<std::string> & output,
           const std::string & separator);

}
