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
#include <hidet/runtime/llm/tokenizer/pattern.h>

RegexPattern::RegexPattern(std::regex pattern) : pattern{std::move(pattern)} {}

std::vector<PatternMatch> RegexPattern::find_matches(const std::string &inside) const {
    auto ibegin = std::sregex_iterator(inside.begin(), inside.end(), pattern);
    auto iend = std::sregex_iterator();

    int prev = 0;
    std::vector<PatternMatch> splits;

    for (auto it = ibegin; it != iend; ++it) {
        int start = static_cast<int>(it->position());
        int n = static_cast<int>(it->length());
        int end = start + n;
        if (prev != start) splits.emplace_back(prev, start, false);
        splits.emplace_back(start, end, true);
        prev = end;
    }

    if (prev != inside.size()) splits.emplace_back(prev, inside.size(), false);

    return splits;
}

std::vector<std::string> Pattern::split(const std::string &s, SplitDelimiterBehavior behavior) const {
    std::vector<PatternMatch> matches = find_matches(s);
    std::vector<std::string> result;

    // Suppose the input string is "abca" and the pattern is "a". Then for each
    // case, the output is...
    for (PatternMatch const &m : matches) {
        std::string segment = s.substr(m.start, m.end - m.start);
        switch (behavior) {
            case SplitDelimiterBehavior::Removed:
                // ["a" "a"]
                if (m.is_match) result.push_back(std::move(segment));
                break;
            case SplitDelimiterBehavior::Isolated:
                // ["a" "bc" "a"]
                result.push_back(std::move(segment));
                break;
            case SplitDelimiterBehavior::MergedWithPrevious:
                // ["abc" "a"]
                if (!m.is_match && !result.empty())
                    result.back() += segment;
                else
                    result.push_back(std::move(segment));
            case SplitDelimiterBehavior::MergedWithNext:
                // ["a" "bca"]
                if (m.is_match && !result.empty())
                    result.back() += segment;
                else
                    result.push_back(std::move(segment));
        }
    }

    return result;
}
