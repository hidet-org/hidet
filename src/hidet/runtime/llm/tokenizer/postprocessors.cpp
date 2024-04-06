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
#include <hidet/runtime/llm/tokenizer/postprocessors.h>

TemplateProcessingPostProcessor::TemplateProcessingPostProcessor(std::vector<std::string> tmpl,
                                                                 std::map<std::string, uint32_t> special_tokens)
    : tmpl{std::move(tmpl)}, special_tokens{std::move(special_tokens)} {
    for (std::string const &s : tmpl) {
        auto it = special_tokens.find(s);
        if (it == special_tokens.end() && s != "A")
            throw std::invalid_argument("TemplateProcessingPostProcessor: unknown piece " + s + " in template");
    }
}

std::vector<uint32_t> TemplateProcessingPostProcessor::process(std::vector<uint32_t> encoding) {
    std::vector<uint32_t> ret;
    for (std::string const &s : tmpl) {
        if (s == "A")
            ret.insert(ret.end(), encoding.begin(), encoding.end());
        else
            ret.push_back(special_tokens.at(s));
    }
    return ret;
}
