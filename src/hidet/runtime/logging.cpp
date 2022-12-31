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


ErrorState* ErrorState::global() {
    static thread_local ErrorState instance;
    return &instance;
}

DLL void hidet_set_last_error(const char *msg) {
    ErrorState* state = ErrorState::global();
    if(state->has_error) {
        fprintf(stderr, "Warning: hidet error state has been override: %s\n", state->error_msg.c_str());
    }
    state->has_error = true;
    state->error_msg = msg;
}

DLL const char * hidet_get_last_error() {
    ErrorState* state = ErrorState::global();
    if(state->has_error) {
        state->has_error = false;
        return state->error_msg.c_str();
    } else {
        return nullptr;
    }
}
