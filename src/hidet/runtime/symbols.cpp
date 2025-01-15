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
#include <hidet/runtime/logging.h>
#include <hidet/runtime/symbols.h>

static std::map<std::string, int32_t> symbol_mapping;

DLL void reset_symbol_table() {
    try {
        symbol_mapping.clear();
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}

DLL int32_t get_symbol_value(const char *symbol_name) {
    try {
        auto it = symbol_mapping.find(symbol_name);
        if (it == symbol_mapping.end()) {
            LOG(ERROR) << "Symbol " << symbol_name << " not found";
        }
        return it->second;
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return 0;
    }
}

DLL void set_symbol_value(const char *symbol_name, int32_t value) {
    try {
        symbol_mapping[symbol_name] = value;
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}
