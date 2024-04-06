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
#include <iomanip>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <hidet/runtime/llm/tokenizer/utf8.h>

// Model takes in a chunk of text from the pre-tokenization step and splits it into a set of tokens,
// each of which are described by their unsigned 32-bit IDs.
class Model {
   public:
    virtual std::vector<uint32_t> tokenize(std::string const &sequence) = 0;
    virtual std::string id_to_token(uint32_t id) = 0;
    virtual ~Model() = default;
};

// BPEWord corresponds roughly to the Hugging Face "Word" implementation. It describes how a chunk of
// text from the pre-tokenization step is split. Initially a chunk of text is split character-wise, and
// the BPE algorithm works to merge these characters into successively larger tokens.
//
// The underlying representation is a doubly-linked list of "Piece" nodes, which point to the previous
// and next token in the sequence. It leverages the fact that the linked list never grows in size to
// avoid dynamic memory allocation entirely -- instead, it uses a fixed-size vector to store the nodes,
// which are then addressed by their index in this vector.
class BPEWord {
   public:
    explicit BPEWord(std::vector<uint32_t> const &ids);

    // merge merges the node at i with the node which follows after it, creating a new node with the given ID.
    int merge(int i, uint32_t new_id);

    // ids provides IDs for all tokens currently in the BPEWord.
    std::vector<uint32_t> ids() const;

    uint32_t at(int i) const { return data[i].id; }
    int prev(int i) const { return data[i].prev; }
    int next(int i) const { return data[i].next; }
    bool valid(int i) const { return data[i].id != std::numeric_limits<uint32_t>::max(); }
    int begin() const { return 0; }
    int end() const { return data.size() - 1; }

   private:
    // Piece corresponds to a single token in the BPEWord. It contains the ID of the token, as well as
    // the "addresses" of the previous and next tokens in the sequence.
    struct Piece {
        // The default ID is the maximum value of a 32-bit unsigned integer, which is also used to invalidate
        // a Piece node and indicate that is no longer in use, as the result of some merges.
        uint32_t id{std::numeric_limits<uint32_t>::max()};
        int prev{};
        int next{};

        Piece() = default;
        Piece(uint32_t id, int prev, int next) : id{id}, prev{prev}, next{next} {};
    };

    // The underlying data structure for the BPEWord.
    std::vector<Piece> data;
};

// BPEModel
class BPEModel: public Model {
   public:
    BPEModel(std::map<std::string, uint32_t> vocab, std::vector<std::pair<std::string, std::string>> const &merges,
             bool byte_fallback);
    std::vector<uint32_t> tokenize(std::string const &sequence) final;
    std::string id_to_token(uint32_t id) final;

   private:
    // The vocabulary that assigns a number to each token.
    std::map<std::string, uint32_t> vocab;
    // Reversed vocabulary, to rebuild sentences.
    std::map<uint32_t, std::string> vocab_r;
    // Contains the mapping between pairs of IDs and their (score, new_id) after
    // merging.
    std::map<std::pair<uint32_t, uint32_t>, std::pair<int, uint32_t>> merges;
    // Caches the results of calls to tokenize.
    std::map<std::string, std::vector<uint32_t>> cache;
    // Whether byte fallbacks (e.g. <0xFA>) should be used for characters not in the vocabulary.
    bool byte_fallback;

    // Helper for tokenize
    void merge_word(BPEWord &word);
};
