from typing import ContextManager
import contextlib
import os
import tempfile


class CapturedStdout:
    def __init__(self):
        self.content: str = ""

    def __str__(self):
        return self.content

    def set_output(self, content: str):
        self.content = content


@contextlib.contextmanager
def capture_stdout() -> ContextManager[CapturedStdout]:
    """
    capture the content that has been printed to stdout in the context

    We did not use `contextlib.redirect_stdout` nor similar functionality in pytest because it does not work with
    `printf(...)` in c/c++.

    usage:
    ```
    with capture_stdout() as captured:
        print("hello world")
    assert captured.content == "hello world\n"
    ```
    """
    captured_stdout = CapturedStdout()

    with tempfile.TemporaryFile(mode='w+') as f:
        new_fd = f.fileno()
        assert new_fd != -1

        original_fd = os.dup(1)
        assert original_fd != -1

        ret = os.dup2(new_fd, 1)
        assert ret != -1

        yield captured_stdout

        ret = os.dup2(original_fd, 1)
        assert ret != -1

        os.close(original_fd)
        f.flush()
        os.fsync(new_fd)
        ret = f.seek(0)
        captured_content = f.read()
        captured_stdout.set_output(captured_content)
