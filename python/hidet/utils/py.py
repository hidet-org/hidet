def prod(seq: list):
    if len(seq) == 0:
        return 1
    else:
        c = seq[0]
        for i in range(1, len(seq)):
            c = c * seq[i]
        return c

