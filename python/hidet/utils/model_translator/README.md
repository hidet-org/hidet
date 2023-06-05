Please see and run example.py for more details
*I would recommend running in a terminal*

Currently, this is a MVP, and many more complicated features of python are not yet supported.

The current approach is to first parse and construct the python ast tree *(see [python docs](https://docs.python.org/3/library/ast.html))*, of either a function or class.
Then with the provided arguments, interpret the ast directly, assigning each node with the runtime type. This is reasonable since DL programs are mostly type stable.
Then with the traced ast, we do other passes to transform any node with the desired type *(ex: torch.softmax)* to the targeted node *(ex: hidet.softmax)*.
For dependencies, we recursively trace, asking for user input if the default filters fail.

However, this means that *(at least as of right now)* this approach does not support many fancy features of python, namely decorators.
The only decorators supported currently are @staticmethod and @property, and implementing both were a pain.

Since this is intended as a tool to aid the model translation process, the errors are explicit, and failure should be visible and left as comments, eg. error handling is a priority.

For future work, it would be more flexible to combine this approach with an architecture similar to the python debugger, involving stack frames together
with the ast, but that sounds like a lot of effort.