# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import ctypes
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Any, List

from transformers import AutoTokenizer

from hidet.ffi.ffi import get_func
from .decoders import _decoder_args
from .models import _model_args
from .normalizers import _normalizer_args
from .postprocessors import _postprocessor_args
from .pretokenizers import _pretokenizer_args
from .utils import _marshal


class TokenizerArgs(ctypes.Structure):
    _fields_ = [
        ("use_normalizer", ctypes.c_bool),
        ("normalizer_type", ctypes.c_char_p),
        ("normalizer_args", ctypes.c_void_p),
        ("use_pretokenizer", ctypes.c_bool),
        ("pretokenizer_type", ctypes.c_char_p),
        ("pretokenizer_args", ctypes.c_void_p),
        ("model_type", ctypes.c_char_p),
        ("model_args", ctypes.c_void_p),
        ("use_postprocessor", ctypes.c_bool),
        ("postprocessor_type", ctypes.c_char_p),
        ("postprocessor_args", ctypes.c_void_p),
        ("decoder_type", ctypes.c_char_p),
        ("decoder_args", ctypes.c_void_p),
    ]


class EncodeResult(ctypes.Structure):
    _fields_ = [
        ("err", ctypes.c_char_p),
        ("dbg", ctypes.c_char_p),
        ("n", ctypes.c_size_t),
        ("data", ctypes.POINTER(ctypes.c_uint32)),
    ]


class DecodeResult(ctypes.Structure):
    _fields_ = [("err", ctypes.c_char_p), ("dbg", ctypes.c_char_p), ("content", ctypes.c_char_p)]


class Tokenizer:
    _ctor = get_func("tokenizer_new", [ctypes.c_void_p], ctypes.c_void_p)
    _dtor = get_func("tokenizer_delete", [ctypes.c_void_p], None)

    _encode = get_func("tokenizer_encode", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_void_p)
    _encode_result_dtor = get_func("encode_result_delete", [ctypes.c_void_p], None)

    _decode = get_func(
        "tokenizer_decode", [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32)], ctypes.c_void_p
    )
    _decode_result_dtor = get_func("decode_result_delete", [ctypes.c_void_p], None)

    def __init__(self, config: Dict[str, Any]):
        args = TokenizerArgs()

        args.use_normalizer = config.get("normalizer") is not None
        if args.use_normalizer:
            args.normalizer_type, args.normalizer_args = _normalizer_args(config["normalizer"])

        args.use_pretokenizer = config.get("pre_tokenizer") is not None
        if args.use_pretokenizer:
            args.pretokenizer_type, args.pretokenizer_args = _pretokenizer_args(config["pre_tokenizer"])

        assert "model" in config
        args.model_type, args.model_args = _model_args(config["model"])

        args.use_postprocessor = config.get("post_processor") is not None
        if args.use_postprocessor:
            args.postprocessor_type, args.postprocessor_args = _postprocessor_args(config["post_processor"])

        assert "decoder" in config
        args.decoder_type, args.decoder_args = _decoder_args(config["decoder"])

        self._tokenizer = Tokenizer._ctor(ctypes.pointer(args))

    def __del__(self):
        Tokenizer._dtor(self._tokenizer)

    def encode(self, text: str) -> List[int]:
        text = _marshal(text)
        res = Tokenizer._encode(self._tokenizer, text)
        with ExitStack() as stack:
            # res should always be freed, even if an exception is raised
            stack.callback(Tokenizer._encode_result_dtor, res)
            res: EncodeResult = ctypes.cast(res, ctypes.POINTER(EncodeResult)).contents
            if res.err is not None:
                raise ValueError(res.err.decode("utf-8"))
            return [res.data[i] for i in range(res.n)]

    def decode(self, tokens: List[int]) -> str:
        tokens = _marshal(tokens, ctypes.POINTER(ctypes.c_uint32))
        res = Tokenizer._decode(self._tokenizer, len(tokens), tokens)
        with ExitStack() as stack:
            # res should always be freed, even if an exception is raised
            stack.callback(Tokenizer._decode_result_dtor, res)
            res: DecodeResult = ctypes.cast(res, ctypes.POINTER(DecodeResult)).contents
            if res.err is not None:
                raise ValueError(res.err.decode("utf-8"))
            return res.content.decode("utf-8")

    @staticmethod
    def from_hugging_face(model: str) -> Tokenizer:
        """
        Create a tokenizer using the tokenizer configuration from Hugging Face.
        """
        tokenizer = AutoTokenizer.from_pretrained(model)
        with tempfile.TemporaryDirectory() as dir:
            tokenizer.save_pretrained(dir)
            with open(Path(dir) / "tokenizer.json") as f:
                config = json.load(f)
        return Tokenizer(config)
