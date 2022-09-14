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

#define API_BEGIN() try {
/*body*/
#define API_END(ret) } catch (const HidetException& e) {          \
                         hidet_set_last_error(e.what());          \
                         return ret;                              \
                     }
