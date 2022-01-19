from typing import List


def doc_join(seq: List, sep):
    rseq = []
    for i in range(len(seq)):
        if i != 0:
            rseq.append(sep)
        rseq.append(seq[i])
    return rseq


class NewLineToken:
    def __init__(self, indent=0):
        self.indent = indent

    def __str__(self):
        return '\n' + ' ' * self.indent


class Doc:
    default_indent = 2

    def __init__(self):
        self.docs = []

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
        for token in self.docs:
            if isinstance(token, NewLineToken):
                token.indent += inc
        return self

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
    def __init__(self, s):
        super().__init__()
        self.docs.append(s)
