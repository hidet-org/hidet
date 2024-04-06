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
#include <hidet/runtime/llm/tokenizer/decoders.h>

SequenceDecoder::SequenceDecoder(std::vector<std::unique_ptr<Decoder>> decoders) : decoders{std::move(decoders)} {}

std::vector<std::string> SequenceDecoder::decode_chain(std::vector<std::string> tokens) {
    for (auto const &decoder : decoders) tokens = decoder->decode_chain(tokens);
    return tokens;
}

ByteLevelDecoder::ByteLevelDecoder() {
    // This is the inverse mapping of ByteLevelPretokenizer: instead of mapping from bytes to a unicode
    // character, we map from a unicode character to a byte.
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174)
            chars_to_bytes[utf8_repr(b)] = b;
        else
            chars_to_bytes[utf8_repr(256 + n++)] = b;
    }
}

ReplaceDecoder::ReplaceDecoder(std::string const &pattern, std::string content)
    : pattern{std::regex{pattern}}, content{std::move(content)} {}

std::vector<std::string> ReplaceDecoder::decode_chain(std::vector<std::string> tokens) {
    std::for_each(tokens.begin(), tokens.end(), [&](std::string &token) {
        std::string new_token;
        for (PatternMatch const &pm : pattern.find_matches(token)) {
            if (pm.is_match)
                new_token += content;
            else
                new_token += token.substr(pm.start, pm.end - pm.start);
        }
        token = new_token;
    });
    return tokens;
}

std::vector<std::string> ByteLevelDecoder::decode_chain(std::vector<std::string> tokens) {
    std::string ret;
    for (std::string const &token : tokens) {
        std::vector<std::string> chrs = utf8_chars(token);
        std::string buf;
        // All or nothing; if even one character is not in the map, we keep the token as-is
        bool ok = true;
        for (std::string &chr : chrs) {
            auto it = chars_to_bytes.find(chr);
            ok &= it != chars_to_bytes.end();
            if (!ok) break;
            buf += static_cast<char>(it->second);
        }
        if (ok)
            ret += buf;
        else
            ret += token;
    }

    // Ensure that the output is valid UTF-8
    return {string_from_utf8_lossy(ret)};
}
std::vector<std::string> FuseDecoder::decode_chain(std::vector<std::string> tokens) {
    std::string ret;
    for (std::string const &token : tokens) {
        ret += token;
    }
    return std::vector<std::string>{ret};
}

StripDecoder::StripDecoder(std::string content, int n_begin, int n_end)
    : content{std::move(content)}, n_begin{n_begin}, n_end{n_end} {}

std::vector<std::string> StripDecoder::decode_chain(std::vector<std::string> tokens) {
    std::for_each(tokens.begin(), tokens.end(), [this](std::string &token) {
        std::vector<std::string> chrs = utf8_chars(token);
        auto start = chrs.begin(), end = chrs.end();
        for (int i = 0; i < n_begin; ++i) {
            if (start != chrs.end() && *start == content)
                ++start;
            else
                break;
        }
        for (int i = 0; i < n_end; ++i) {
            if (end != chrs.begin() && *(end - 1) == content)
                --end;
            else
                break;
        }
        token.clear();
        for (auto it = start; it != end; ++it) token += *it;
    });
    return tokens;
}

std::vector<std::string> ByteFallbackDecoder::decode_chain(std::vector<std::string> tokens) {
    std::vector<std::string> new_tokens;

    std::string buf;
    auto commitBuf = [&]() {
        std::vector<std::string> chrs = utf8_chars(buf);
        bool valid = std::all_of(chrs.begin(), chrs.end(), utf8_is_valid);

        // If the buffer is a valid UTF-8 string, commit it, otherwise use the replacement
        // character U+FFFD for each invalid byte.
        if (valid)
            new_tokens.push_back(std::move(buf));
        else {
            for (size_t i = 0; i < buf.size(); ++i) {
                new_tokens.emplace_back("\xEF\xBF\xBD");
            }
        }
        buf.clear();
    };

    for (std::string const &token : tokens) {
        if (token.size() == 6 && token.substr(0, 3) == "<0x" && *token.rbegin() == '>') {
            // Some byte fallback character of the form <0xAB>
            std::stringstream tmp{token.substr(3, 2)};
            uint32_t byte;  // operator>> only reads one character for uint8_t
            tmp >> std::hex >> byte;
            buf += static_cast<char>(byte);
        } else {
            // Some character that isn't a byte fallback
            commitBuf();
            new_tokens.emplace_back(token);
        }
    }
    commitBuf();

    return new_tokens;
}
