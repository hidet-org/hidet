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
#include <hidet/runtime/llm/tokenizer/normalizers.h>

SequenceNormalizer::SequenceNormalizer(std::vector<std::unique_ptr<Normalizer>> normalizers)
    : normalizers{std::move(normalizers)} {}

void SequenceNormalizer::normalize(std::string &s) {
    for (auto const &normalizer : normalizers) normalizer->normalize(s);
}

PrependNormalizer::PrependNormalizer(std::string prefix) : prefix{std::move(prefix)} {}

void PrependNormalizer::normalize(std::string &s) {
    if (!s.empty()) s = prefix + s;
}

ReplaceNormalizer::ReplaceNormalizer(const std::string &pattern, std::string content)
    : pattern{std::regex{pattern}}, content{std::move(content)} {}

void ReplaceNormalizer::normalize(std::string &s) {
    std::string res;
    for (PatternMatch const &pm : pattern.find_matches(s)) {
        if (pm.is_match)
            res += content;
        else
            res += s.substr(pm.start, pm.end - pm.start);
    }
    s = std::move(res);
}
