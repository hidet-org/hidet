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
#include <cstddef>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <cstring>
#include <vector>
#include <hidet/runtime/llm/tokenizer/pattern.h>
#include <hidet/runtime/llm/tokenizer/utf8.h>

// Decoder manipulates a sequence of raw tokens to produce a human-readable form.
class Decoder {
   public:
    virtual std::vector<std::string> decode_chain(std::vector<std::string> tokens) = 0;
    virtual ~Decoder() = default;

    std::string decode(std::vector<std::string> tokens) {
        std::string ret;
        for (std::string const &s : decode_chain(std::move(tokens))) ret += s;
        return ret;
    }
};

// SequenceDecoder runs a sequence of decoders sequentially, passing the output of one decoder into another.
class SequenceDecoder: public Decoder {
   public:
    explicit SequenceDecoder(std::vector<std::unique_ptr<Decoder>> decoders);
    std::vector<std::string> decode_chain(std::vector<std::string> tokens) final;

   private:
    std::vector<std::unique_ptr<Decoder>> decoders;
};

// ReplaceDecoder replaces all instances of a pattern with another on a token-by-token basis.
class ReplaceDecoder: public Decoder {
   public:
    explicit ReplaceDecoder(std::string const &pattern, std::string content);
    std::vector<std::string> decode_chain(std::vector<std::string> tokens) final;

   private:
    RegexPattern pattern;
    std::string content;
};

// ByteLevelDecoder is meant to be used with ByteLevel pre-tokenization (for example, in GPT-2). It reverses the
// mapping of the ByteLevel pre-tokenization step, converting the byte-level replacements back into human-readable
// characters.
class ByteLevelDecoder: public Decoder {
   public:
    ByteLevelDecoder();
    std::vector<std::string> decode_chain(std::vector<std::string> tokens) final;

   private:
    std::map<std::string, uint8_t> chars_to_bytes;
};

class FuseDecoder: public Decoder {
   public:
    std::vector<std::string> decode_chain(std::vector<std::string> tokens) final;
};

class StripDecoder: public Decoder {
    std::string content;
    int n_begin;
    int n_end;

   public:
    explicit StripDecoder(std::string content, int n_begin, int n_end);
    std::vector<std::string> decode_chain(std::vector<std::string> tokens) final;
};

class ByteFallbackDecoder: public Decoder {
   public:
    std::vector<std::string> decode_chain(std::vector<std::string> tokens) final;
};
