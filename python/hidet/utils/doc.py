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
from typing import List, Union


def doc_join(seq: List, sep):
    doc = Doc()
    for i in range(len(seq)):
        if i != 0:
            doc += sep
        doc += seq[i]
    return doc


class NewLineToken:
    def __init__(self, indent=0):
        self.indent = indent

    def __str__(self):
        return '\n' + ' ' * self.indent


class Doc:
    default_indent = 2

    def __init__(self):
        self.docs: List[Union[str, NewLineToken]] = []

    def append(self, doc):
        if isinstance(doc, list):
            for item in doc:
                self.append(item)
        elif isinstance(doc, Doc):
            self.docs.extend(doc.docs)
        elif isinstance(doc, str):
            self.docs.append(doc)
        else:
            raise NotImplementedError()

    def indent(self, inc=None):
        if inc is None:
            inc = self.default_indent
        doc = Doc()
        for token in self.docs:
            if isinstance(token, NewLineToken):
                doc.docs.append(NewLineToken(indent=token.indent + inc))
            else:
                doc.docs.append(token)
        return doc

    def trim(self):
        i = 0
        while i < len(self.docs) and isinstance(self.docs[i], NewLineToken):
            i += 1
        j = len(self.docs)
        while j > i and isinstance(self.docs[j - 1], NewLineToken):
            j -= 1
        doc = Doc()
        for k in range(i, j):
            token = self.docs[k]
            if isinstance(token, NewLineToken):
                doc.docs.append(NewLineToken(indent=token.indent))
            else:
                doc.docs.append(token)
        return doc

    def __add__(self, other):
        doc = Doc()
        doc.docs = [token for token in self.docs]
        doc += other
        return doc

    def __radd__(self, other):
        doc = Doc()
        doc.docs = []
        doc.append(other)
        doc.append(self)
        return doc

    def __iadd__(self, other):
        self.append(other)
        return self

    def __str__(self):
        return "".join(str(s) for s in self.docs)


class NewLine(Doc):
    def __init__(self, indent=0):
        super().__init__()
        self.docs.append(NewLineToken(indent))


class Text(Doc):
    def __init__(self, s: str):
        super().__init__()
        assert isinstance(s, str)
        self.docs.append(s)
        self.format_str = s

    def format(self, *args) -> Doc:
        format_str: str = self.format_str
        texts = format_str.split('{}')
        if len(texts) != len(args) + 1:
            raise ValueError(f'format string {format_str} does not match the number of args: {len(args)}')
        return doc_join([Text(texts[i]) + args[i] for i in range(len(args))], "") + Text(texts[-1])
