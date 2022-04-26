from __future__ import annotations


class A:
    @staticmethod
    def get() -> A:
        return A()


def main():
    A()
    a = A.get()


if __name__ == '__main__':
    main()

