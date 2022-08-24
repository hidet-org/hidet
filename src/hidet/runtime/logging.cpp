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
