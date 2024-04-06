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
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <hidet/runtime/llm/tokenizer/pattern.h>

// PreTokenizer takes a vector of strings (usually a single string) and splits it
// into smaller chunks, allowing for more efficient tokenization. This is the
// second step in the tokenization pipeline after normalization and before tokenization.
class PreTokenizer {
   public:
    virtual void pre_tokenize(std::vector<std::string> &pretokenized) = 0;
    virtual ~PreTokenizer() = default;
};

// ByteLevelPreTokenizer is a PreTokenizer that splits a string based on a predefined
// regex pattern and maps every byte to a unique unicode character.
class ByteLevelPreTokenizer: public PreTokenizer {
    // Whether to add a leading space to the first word. This allows to treat the
    // leading word just as any other word.
    bool add_prefix_space;

    // Whether to use the standard GPT-2 regex for whitespace splitting.
    bool use_regex;

    // Mapping of raw bytes (0x00 - 0xff) to unicode character replacements.
    std::vector<std::string> bytes_to_chars;

   public:
    explicit ByteLevelPreTokenizer(bool add_prefix_space, bool use_regex);
    void pre_tokenize(std::vector<std::string> &pretokenized) final;
};
