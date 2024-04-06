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
#include <regex>
#include <string>
#include <vector>

// SplitDelimiterBehavior defines how the delimiter is treated when splitting a string.
enum class SplitDelimiterBehavior {
    Removed,             // A.B -> [A B]
    Isolated,            // A.B -> [A . B]
    MergedWithPrevious,  // A.B -> [A. B]
    MergedWithNext,      // A.B -> [A .B]
};

// PatternMatch represent a segment of a string that is either a match or not.
struct PatternMatch {
    // The starting index of the segment, inclusive.
    int start;
    // The ending index of the segment, exclusive.
    int end;
    // Whether the segment is a match or not.
    bool is_match;

    PatternMatch(int start, int end, bool is_match) : start{start}, end{end}, is_match{is_match} {}
};

class Pattern {
   public:
    // Suppose the string has length n, report a partition of [0, n) into intervals that are all either match/no match.
    // For example with inside = "abaca" and the pattern "ba", the result is [0 1 F], [1 3 T], [3 5 F].
    virtual std::vector<PatternMatch> find_matches(std::string const &inside) const = 0;

    std::vector<std::string> split(const std::string &s, SplitDelimiterBehavior behaviour) const;

    virtual ~Pattern() = default;
};

class RegexPattern: public Pattern {
    std::regex pattern;

   public:
    explicit RegexPattern(std::regex pattern);
    std::vector<PatternMatch> find_matches(std::string const &inside) const final;
};
