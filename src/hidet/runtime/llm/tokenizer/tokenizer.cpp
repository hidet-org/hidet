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

#include <hidet/runtime/common.h>
#include <hidet/runtime/llm/tokenizer/debug.h>
#include <hidet/runtime/llm/tokenizer/decoders.h>
#include <hidet/runtime/llm/tokenizer/models.h>
#include <hidet/runtime/llm/tokenizer/normalizers.h>
#include <hidet/runtime/llm/tokenizer/postprocessors.h>
#include <hidet/runtime/llm/tokenizer/pretokenizers.h>

// Extern global debugging stream declared here
std::stringstream dbg;

// Normalizer ==========================================================================================================

struct SequenceNormalizerArgs {
    size_t n;
    char **types;
    void **child_args;
};

struct PrependNormalizerArgs {
    char *prefix;
};

struct ReplaceNormalizerArgs {
    char *from;
    char *to;
};

Normalizer *make_normalizer(char *type, void *args_vp) {  // NOLINT(misc-no-recursion)
    // SequenceNormalizer
    if (std::string{type} == "Sequence") {
        auto args = static_cast<SequenceNormalizerArgs *>(args_vp);
        std::vector<std::unique_ptr<Normalizer>> normalizers;
        normalizers.reserve(args->n);
        for (size_t i = 0; i < args->n; i++) {
            Normalizer *n = make_normalizer(args->types[i], args->child_args[i]);
            normalizers.emplace_back(n);
        }
        return new SequenceNormalizer(std::move(normalizers));
    }

    // PrependNormalizer
    else if (std::string{type} == "Prepend") {
        auto args = static_cast<PrependNormalizerArgs *>(args_vp);
        return new PrependNormalizer(args->prefix);
    }

    // ReplaceNormalizer
    else if (std::string{type} == "Replace") {
        auto args = static_cast<ReplaceNormalizerArgs *>(args_vp);
        return new ReplaceNormalizer(args->from, args->to);
    }

    throw std::invalid_argument("Unknown normalizer type: " + std::string{type});
}

// PreTokenizer ========================================================================================================

struct ByteLevelPreTokenizerArgs {
    bool add_prefix_space;
    bool use_regex;
};

PreTokenizer *make_pretokenizer(char *type, void *args_vp) {
    // ByteLevelPreTokenizer
    if (std::string{type} == "ByteLevel") {
        auto args = static_cast<ByteLevelPreTokenizerArgs *>(args_vp);
        return new ByteLevelPreTokenizer(args->add_prefix_space, args->use_regex);
    }

    throw std::invalid_argument("Unknown pretokenizer type: " + std::string{type});
}

// Model ===============================================================================================================

struct BPEModelArgs {
    struct VocabEntry {
        char *token;
        uint32_t id;
    };
    struct MergeEntry {
        char *first;
        char *second;
    };

    size_t n_vocab;
    VocabEntry *vocab;
    size_t n_merges;
    MergeEntry *merges;

    bool byte_fallback;
};

Model *make_model(char *type, void *args_vp) {
    // BPEModel
    if (std::string{type} == "BPE") {
        auto args = static_cast<BPEModelArgs *>(args_vp);
        std::map<std::string, uint32_t> vocab;
        for (int i = 0; i < args->n_vocab; i++) vocab[args->vocab[i].token] = args->vocab[i].id;
        std::vector<std::pair<std::string, std::string>> merges;
        merges.reserve(args->n_merges);
        for (int i = 0; i < args->n_merges; i++) merges.emplace_back(args->merges[i].first, args->merges[i].second);
        return new BPEModel(vocab, merges, args->byte_fallback);
    }

    throw std::invalid_argument("Unknown model type: " + std::string{type});
}

// Postprocessor =======================================================================================================

struct TemplateProcessingPostProcessorArgs {
    struct SpecialTokenEntry {
        char *token;
        uint32_t id;
    };

    size_t n_tmpl;
    char **tmpl;
    size_t n_special_tokens;
    SpecialTokenEntry *special_tokens;
};

PostProcessor *make_postprocessor(char *type, void *args_vp) {
    // ByteLevelPostProcessor
    if (std::string{type} == "ByteLevel") {
        // ByteLevelPostProcessor is a no-op for our case
        return new ByteLevelPostProcessor{};
    }
    // TemplateProcessingPostProcessor
    if (std::string{type} == "TemplateProcessing") {
        auto args = static_cast<TemplateProcessingPostProcessorArgs *>(args_vp);
        std::vector<std::string> tmpl(args->n_tmpl);
        std::copy(args->tmpl, args->tmpl + args->n_tmpl, tmpl.begin());
        std::map<std::string, uint32_t> special_tokens;
        for (int i = 0; i < args->n_special_tokens; i++)
            special_tokens[args->special_tokens[i].token] = args->special_tokens[i].id;
        return new TemplateProcessingPostProcessor(tmpl, special_tokens);
    }

    throw std::invalid_argument("Unknown postprocessor type: " + std::string{type});
}

// Decoder =============================================================================================================

struct SequenceDecoderArgs {
    size_t n;
    char **types;
    void **child_args;
};

struct ReplaceDecoderArgs {
    char *pattern;
    char *content;
};

struct StripDecoderArgs {
    char *content;
    int n_begin;
    int n_end;
};

Decoder *make_decoder(char *type, void *args_vp) {  // NOLINT(misc-no-recursion)
    // SequenceDecoder
    if (std::string{type} == "Sequence") {
        auto args = static_cast<SequenceDecoderArgs *>(args_vp);
        std::vector<std::unique_ptr<Decoder>> decoders;
        decoders.reserve(args->n);
        for (size_t i = 0; i < args->n; i++) {
            Decoder *d = make_decoder(args->types[i], args->child_args[i]);
            decoders.emplace_back(d);
        }
        return new SequenceDecoder(std::move(decoders));
    }
    // ReplaceDecoder
    if (std::string{type} == "Replace") {
        auto args = static_cast<ReplaceDecoderArgs *>(args_vp);
        return new ReplaceDecoder(args->pattern, args->content);
    }
    // ByteLevelDecoder
    if (std::string{type} == "ByteLevel") {
        return new ByteLevelDecoder{};
    }
    // FuseDecoder
    if (std::string{type} == "Fuse") {
        return new FuseDecoder{};
    }
    // StripDecoder
    if (std::string{type} == "Strip") {
        auto args = static_cast<StripDecoderArgs *>(args_vp);
        return new StripDecoder(args->content, args->n_begin, args->n_end);
    }
    // ByteFallbackDecoder
    if (std::string{type} == "ByteFallback") {
        return new ByteFallbackDecoder{};
    }

    throw std::invalid_argument("Unknown decoder type: " + std::string{type});
}

// Tokenizer ===========================================================================================================

struct Tokenizer {
    std::unique_ptr<Normalizer> normalizer;
    std::unique_ptr<PreTokenizer> pretokenizer;
    std::unique_ptr<Model> model;
    std::unique_ptr<PostProcessor> postprocessor;
    std::unique_ptr<Decoder> decoder;

    Tokenizer(std::unique_ptr<Normalizer> normalizer, std::unique_ptr<PreTokenizer> pretokenizer,
              std::unique_ptr<Model> model, std::unique_ptr<PostProcessor> postprocessor,
              std::unique_ptr<Decoder> decoder)
        : normalizer(std::move(normalizer)),
          pretokenizer(std::move(pretokenizer)),
          model(std::move(model)),
          postprocessor(std::move(postprocessor)),
          decoder(std::move(decoder)) {}

    std::vector<uint32_t> encode(std::string s) const {
        if (normalizer) normalizer->normalize(s);

        std::vector<std::string> pretokens{s};
        if (pretokenizer) pretokenizer->pre_tokenize(pretokens);

        std::vector<uint32_t> tokens;
        for (std::string const &piece : pretokens)
            for (uint32_t id : model->tokenize(piece)) tokens.push_back(id);

        if (postprocessor) tokens = postprocessor->process(tokens);

        return tokens;
    }

    std::string decode(std::vector<uint32_t> const &ids) const {
        std::vector<std::string> tokens;
        tokens.reserve(ids.size());
        for (uint32_t id : ids) tokens.push_back(model->id_to_token(id));
        return decoder->decode(tokens);
    }
};

struct TokenizerArgs {
    bool use_normalizer;
    char *normalizer_type;
    void *normalizer_args;

    bool use_pretokenizer;
    char *pretokenizer_type;
    void *pretokenizer_args;

    char *model_type;
    void *model_args;

    bool use_postprocessor;
    char *postprocessor_type;
    void *postprocessor_args;

    char *decoder_type;
    void *decoder_args;
};

DLL void *tokenizer_new(void *args_vp) {
    auto args = static_cast<TokenizerArgs *>(args_vp);

    std::unique_ptr<Normalizer> normalizer;
    if (args->use_normalizer) {
        normalizer = std::unique_ptr<Normalizer>(make_normalizer(args->normalizer_type, args->normalizer_args));
    }

    std::unique_ptr<PreTokenizer> pretokenizer;
    if (args->use_pretokenizer) {
        pretokenizer =
            std::unique_ptr<PreTokenizer>(make_pretokenizer(args->pretokenizer_type, args->pretokenizer_args));
    }

    auto model = std::unique_ptr<Model>(make_model(args->model_type, args->model_args));

    std::unique_ptr<PostProcessor> postprocessor;
    if (args->use_postprocessor) {
        postprocessor =
            std::unique_ptr<PostProcessor>(make_postprocessor(args->postprocessor_type, args->postprocessor_args));
    }

    auto decoder = std::unique_ptr<Decoder>(make_decoder(args->decoder_type, args->decoder_args));

    return new Tokenizer(std::move(normalizer), std::move(pretokenizer), std::move(model), std::move(postprocessor),
                         std::move(decoder));
}

DLL void tokenizer_delete(void *tokenizer_vp) {
    auto tokenizer = static_cast<Tokenizer *>(tokenizer_vp);
    delete tokenizer;
}

struct EncodeResult {
    char *err;
    char *dbg;

    size_t n;
    uint32_t *ids;

    ~EncodeResult() {
        delete[] err;
        delete[] dbg;
        delete[] ids;
    }
};

DLL void encode_result_delete(void *result_vp) {
    auto result = static_cast<EncodeResult *>(result_vp);
    delete result;
}

DLL void *tokenizer_encode(void *tokenizer_vp, char *s) {
    char *debug = strdup(dbg.str().c_str());
    dbg.clear();

    try {
        auto tokenizer = static_cast<Tokenizer *>(tokenizer_vp);
        auto tmp = tokenizer->encode(s);
        auto ids = new uint32_t[tmp.size()];
        std::copy(tmp.begin(), tmp.end(), ids);
        return new EncodeResult{nullptr, debug, tmp.size(), ids};
    } catch (const std::exception &e) {
        return new EncodeResult{strdup(e.what()), debug, 0, nullptr};
    }
}

struct DecodeResult {
    char *err;
    char *dbg;
    char *content;

    ~DecodeResult() {
        delete[] err;
        delete[] dbg;
        delete[] content;
    }
};

DLL void decode_result_delete(void *result_vp) {
    auto result = static_cast<DecodeResult *>(result_vp);
    delete result;
}

DLL void *tokenizer_decode(void *tokenizer_vp, size_t n, uint32_t *ids) {
    char *debug = strdup(dbg.str().c_str());
    try {
        auto tokenizer = static_cast<Tokenizer *>(tokenizer_vp);
        std::string res = tokenizer->decode(std::vector<uint32_t>(ids, ids + n));
        return new DecodeResult{nullptr, debug, strdup(res.c_str())};
    } catch (const std::exception &e) {
        return new DecodeResult{strdup(e.what()), debug, nullptr};
    }
}
