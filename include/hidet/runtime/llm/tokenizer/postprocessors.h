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
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// PostProcessor takes the output of the BPE algorithm (a sequence of token IDs) and applies a transformation to it
// to prepare it for consumption by a language model. Usually, this involves prepending/appending special
// starting/ending tokens.
class PostProcessor {
   public:
    virtual std::vector<uint32_t> process(std::vector<uint32_t> encoding) = 0;
    virtual ~PostProcessor() = default;
};

// ByteLevelPostProcessor is a no-op post-processor that returns the input encoding as-is. The provided functionality
// by Hugging Face maps IDs back to their source offsets, which we don't need to do here.
class ByteLevelPostProcessor: public PostProcessor {
   public:
    std::vector<uint32_t> process(std::vector<uint32_t> encoding) final { return encoding; };
};

// TemplateProcessingPostProcessor is a post-processor that takes a vector of strings (called the "template"), which
// define the output of the model. The template is a list of strings, where each string is either a special token or
// "A". If the string is "A", the output of the model is inserted into the output. If the string is a special token,
// the ID of the special token is inserted into the output.
class TemplateProcessingPostProcessor: public PostProcessor {
    std::vector<std::string> tmpl;
    std::map<std::string, uint32_t> special_tokens;

   public:
    TemplateProcessingPostProcessor(std::vector<std::string> tmpl, std::map<std::string, uint32_t> special_tokens);
    std::vector<uint32_t> process(std::vector<uint32_t> encoding) final;
};
