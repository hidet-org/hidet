// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <iostream>
#include <sstream>
#include <hidet/runtime/common.h>

struct ErrorState {
    bool has_error;
    std::string error_msg;

    static ErrorState *global();
};

struct HidetException: std::exception {
    std::string file;
    int line;
    std::string msg;

    HidetException(std::string file, int line, std::string msg):file(file), line(line), msg(msg){}

    const char * what() const noexcept override {
        static std::string what_msg;
        what_msg = this->file + ":" + std::to_string(this->line) + " " + this->msg;
        return what_msg.c_str();
    }
};

DLL void hidet_set_last_error(const char *msg);

DLL const char * hidet_get_last_error();
