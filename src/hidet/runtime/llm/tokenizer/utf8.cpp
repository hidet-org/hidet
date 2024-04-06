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
#include <hidet/runtime/llm/tokenizer/utf8.h>

std::vector<std::string> utf8_chars(std::string const &input) {
    std::string buf;
    std::vector<std::string> ret;

    for (char c : input) {
        bool is_continuation = (c & 0xc0) == 0x80;
        if (!is_continuation && !buf.empty()) {
            ret.emplace_back(std::move(buf));
            buf.clear();
        }
        buf += c;
    }
    if (!buf.empty()) {
        ret.emplace_back(std::move(buf));
        buf.clear();
    }
    return ret;
}

bool utf8_is_valid(std::string const &input) {
    if (input.empty() || input.size() > 4) return false;

    uint32_t repr = 0;
    for (char c : input) repr = (repr << 8) | static_cast<uint8_t>(c);

    // https://stackoverflow.com/questions/66715611/check-for-valid-utf-8-encoding-in-c
    if (repr <= 0x7f)
        return true;
    else if (0xc280 <= repr && repr <= 0xdfbf)
        return ((repr & 0xe0c0) == 0xc080);
    else if (0xeda080 <= repr && repr <= 0xedbfbf)
        return false;  // Reject UTF-16 surrogates
    else if (0xe0a080 <= repr && repr <= 0xefbfbf)
        return ((repr & 0xf0c0c0) == 0xe08080);
    else if (0xf0908080 <= repr && repr <= 0xf48fbfbf)
        return ((repr & 0xf8c0c0c0) == 0xf0808080);
    else
        return false;
}

std::string utf8_repr(uint32_t codepoint) {
    std::string ret;
    if (codepoint <= 0x7f) {
        // 0xxxxxxx
        ret += static_cast<char>(codepoint);
    } else if (codepoint <= 0x7ff) {
        // 110xxxxx 10xxxxxx
        ret += static_cast<char>(0xc0 | (codepoint >> 6));
        ret += static_cast<char>(0x80 | (codepoint & 0x3f));
    } else if (codepoint <= 0xffff) {
        // 1110xxxx 10xxxxxx 10xxxxxx
        ret += static_cast<char>(0xe0 | (codepoint >> 12));
        ret += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
        ret += static_cast<char>(0x80 | (codepoint & 0x3f));
    } else {
        // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        ret += static_cast<char>(0xf0 | (codepoint >> 18));
        ret += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
        ret += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
        ret += static_cast<char>(0x80 | (codepoint & 0x3f));
    }
    return ret;
}

std::string string_from_utf8_lossy(std::string const &input) {
    std::string ret;
    ret.reserve(input.size());
    for (std::string const &chr : utf8_chars(input)) {
        if (utf8_is_valid(chr))
            ret += chr;
        else
            ret += "\xEF\xBF\xBD";  // U+FFFD replacement character
    }
    return ret;
}
