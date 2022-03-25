def normalize(v, num=2):
    if isinstance(v, (list, tuple)):
        return v
    else:
        return [v for _ in range(num)]
