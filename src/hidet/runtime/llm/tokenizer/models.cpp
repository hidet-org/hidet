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
#include <hidet/runtime/llm/tokenizer/models.h>

BPEModel::BPEModel(std::map<std::string, uint32_t> vocab,
                   const std::vector<std::pair<std::string, std::string>> &merges, bool byte_fallback)
    : vocab{std::move(vocab)}, byte_fallback{byte_fallback} {
    // Initialize vocab_r from vocab
    for (auto const &item : this->vocab) {
        std::string k = item.first;
        uint32_t v = item.second;
        vocab_r[v] = std::move(k);
    }

    // Initialize merge map
    for (int i = 0; i < merges.size(); ++i) {
        std::string const &a = merges[i].first, &b = merges[i].second;
        uint32_t a_id = this->vocab.at(a), b_id = this->vocab.at(b);
        uint32_t new_id = this->vocab.at(a + b);
        this->merges[{a_id, b_id}] = {i, new_id};
    }
}

std::vector<uint32_t> BPEModel::tokenize(const std::string &sequence) {
    if (cache.find(sequence) != cache.end()) {
        return cache[sequence];
    }

    // Hugging Face has more complicated logic here around padding the input with continuing_subword_prefix /
    // end_of_word_suffix, which we don't need for now: LLaMA provides byte fallbacks in the vocabulary, and GPT-2/OPT
    // handle bad characters in the pre-tokenization step.

    // 1. Split the input sequence into Unicode characters.
    std::vector<uint32_t> ids;
    for (std::string const &chr : utf8_chars(sequence)) {
        auto it = vocab.find(chr);
        if (it != vocab.end()) {
            // This is the easy case; we have already an ID for this Unicode character.
            ids.push_back(it->second);
        } else {
            // Here, we are trying to find the ID of some Unicode character that is not in the vocabulary. Our only
            // option at this point is to hope that byte fallback is enabled, in which case we map the character to
            // its individual bytes -- for example, the character "é" would map to <0xC3><0xA9>.
            if (!byte_fallback) throw std::invalid_argument("Unknown character (byte fallback is disabled): " + chr);
            for (char c : chr) {
                std::stringstream ss;
                uint8_t byte = c;
                ss << std::hex << std::uppercase << std::setfill('0');
                ss << "<0x" << std::setw(2) << +byte << ">";
                ids.push_back(vocab.at(ss.str()));
            }
        }
    }

    // 2. Apply merges to the sequence of Unicode characters.
    BPEWord word{ids};
    merge_word(word);
    return cache[sequence] = word.ids();
}

std::string BPEModel::id_to_token(uint32_t id) {
    return vocab_r.at(id);
}

void BPEModel::merge_word(BPEWord &word) {
    // pq is a priority queue of (score, location, target_id) tuples, which encode candidates for the next merge to
    // apply. Candidates for merges may be invalidated by other merges, so before applying merges we should check that
    // they are still valid.
    std::priority_queue<std::tuple<int, int, uint32_t>> pq;
    auto push = [&](int i) {
        int j = word.next(i);
        auto it = merges.find({word.at(i), word.at(j)});
        if (it != merges.end()) {
            int score = it->second.first;
            uint32_t target_id = it->second.second;
            pq.emplace(-score, i, target_id);
        }
    };

    // To begin, our set of candidates is all adjacent pairs of characters.
    int first = word.begin(), second = word.next(first);
    while (second != word.end()) {
        push(first);
        first = second;
        second = word.next(second);
    }

    // Merge until there's nothing left to merge; select merges greedily based on the score.
    while (!pq.empty()) {
        auto top = pq.top();
        int i = std::get<1>(top);
        uint32_t target_id = std::get<2>(top);
        pq.pop();

        // Invalidation check 1: index i might have been merged into another piece which starts before index i.
        if (!word.valid(i)) continue;

        // Invalidation check 2: we know at this point that there's still a piece at index i, but maybe it has been
        // merged into the piece which came after it at the time we pushed this entry. We can check this by checking
        // that merging the piece starting at i with the piece starting at j still produces the same token.
        int j = word.next(i);
        auto it = merges.find({word.at(i), word.at(j)});
        if (it == merges.end() || it->second.second != target_id) continue;

        // Merge the piece starting at i with the piece starting at j. After merging, we have two new candidates to
        // consider.
        uint32_t new_id = it->second.second;
        word.merge(i, new_id);
        if (i != word.begin()) push(word.prev(i));
        if (word.next(i) != word.end()) push(i);
    }
}

BPEWord::BPEWord(std::vector<uint32_t> const &ids) : data(ids.size() + 1) {
    size_t n = ids.size();
    for (int i = 0; i <= n; ++i) {
        // We pad an extra element at the end to avoid some boundary checks later.
        uint32_t id = i == n ? std::numeric_limits<uint32_t>::max() : ids[i];
        data[i] = {id, i - 1, i + 1};
    }
}

int BPEWord::merge(int i, uint32_t new_id) {
    // Merge the pair of tokens starting at index i into a new ID.
    // Before: i <-> j <-> k
    // After: i <-> k
    int j = next(i), k = next(j);
    data[i].id = new_id;
    data[i].next = k;
    data[k].prev = i;
    data[j].id = std::numeric_limits<uint32_t>::max();  // Invalidate index j
    return i;
}

std::vector<uint32_t> BPEWord::ids() const {
    std::vector<uint32_t> ret;
    for (int i = 0; i != end(); i = next(i)) ret.push_back(data[i].id);
    return ret;
}
