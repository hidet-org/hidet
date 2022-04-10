class Apple:
    def __init__(self, name):
        self.name = name
        print('init ' + self.name)

    def __del__(self):
        print('del ' + self.name)


class Orange:
    def __init__(self, name):
        self.dct = dict()
        self.dct['a'] = []
        self.dct['a'].append(Apple('xxx'))


    def doit(self):
        return self.dct['a'].pop()


if __name__ == '__main__':
    a = Apple('a')
    del a
    b = Orange('c')
    c = b.doit()
    del c
    print('here')

