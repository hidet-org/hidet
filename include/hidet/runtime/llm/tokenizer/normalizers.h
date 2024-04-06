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
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <hidet/runtime/llm/tokenizer/pattern.h>

// Normalizer takes in an input text and applies a transformation to normalize it.
// This is the first step in the tokenization pipeline, coming before the pre-tokenizer.
class Normalizer {
   public:
    virtual void normalize(std::string &s) = 0;
    virtual ~Normalizer() = default;
};

// SequenceNormalizer runs a pre-defined set of normalizers in sequence.
class SequenceNormalizer: public Normalizer {
    std::vector<std::unique_ptr<Normalizer>> normalizers;

   public:
    explicit SequenceNormalizer(std::vector<std::unique_ptr<Normalizer>> normalizers);
    void normalize(std::string &s) final;
};

// PrependNormalizer prepends a prefix to a string.
class PrependNormalizer: public Normalizer {
    std::string prefix;

   public:
    explicit PrependNormalizer(std::string prefix);
    void normalize(std::string &s) final;
};

// ReplaceNormalizer replaces all instances of a pattern with the given content.
class ReplaceNormalizer: public Normalizer {
    RegexPattern pattern;
    std::string content;

   public:
    explicit ReplaceNormalizer(const std::string &pattern, std::string content);
    void normalize(std::string &s) final;
};
