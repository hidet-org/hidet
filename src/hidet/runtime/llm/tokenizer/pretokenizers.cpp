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
#include <hidet/runtime/llm/tokenizer/pretokenizers.h>
#include <hidet/runtime/llm/tokenizer/utf8.h>

void ByteLevelPreTokenizer::pre_tokenize(std::vector<std::string> &pretokenized) {
    // This is an approximation of the GPT-2 regex, which requires Unicode property escapes currently not supported
    // by std::regex.
    static std::regex re(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+(?!\S)|\s+)");
    auto pat = RegexPattern{re};

    // 1. Pre-tokenize with regex
    std::vector<std::string> res;
    res.reserve(pretokenized.size());
    for (std::string const &s : pretokenized) {
        if (add_prefix_space && s[0] != ' ') res.push_back(" " + s);
        if (use_regex) {
            std::vector<std::string> splits = pat.split(s, SplitDelimiterBehavior::Isolated);
            res.insert(res.end(), splits.begin(), splits.end());
        }
    }

    // 2. Map each byte to a Unicode character
    std::for_each(res.begin(), res.end(), [this](std::string &s) {
        std::string ret;
        for (char c : s) ret += bytes_to_chars[static_cast<uint8_t>(c)];
        s = std::move(ret);
    });

    pretokenized = res;
}

ByteLevelPreTokenizer::ByteLevelPreTokenizer(bool add_prefix_space, bool use_regex)
    : add_prefix_space{add_prefix_space}, use_regex{use_regex}, bytes_to_chars(256) {
    // Precompute the mapping of bytes to unicode characters. This is derived from the GPT-2 byte-level BPE
    // implementation here:
    // https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174)
            bytes_to_chars[b] = utf8_repr(b);
        else
            bytes_to_chars[b] = utf8_repr(256 + n++);
    }
}
