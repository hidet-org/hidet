class DynamoConfig:
    def __init__(self):
        self._search_space: int = 0
        self._parallel_k: str = 'default'
        self._use_fp16: bool = False
        self._use_fp16_reduction: bool = False
        self._use_cuda_graph: bool = True
        self._print_input_graph: bool = False
        self._correctness_report: bool = False

    def __getitem__(self, item: str):
        assert isinstance(item, str)
        return getattr(self, f"_{item}")

    def search_space(self, level: int = 2):
        """
        The schedule search space for the operator kernel tuning
        Candidates are: 0, 1, 2
         - 0: Use the default schedule, without tuning.
         - 1: Tune the schedule in a small search space. Usually takes less than one minute to tune a kernel.
         - 2: Tune the schedule in a large search space. Usually achieves the best performance, but takes longer time.
        """
        self._search_space = level
        return self

    def parallel_k(self, strategy="default"):
        """
        Parallelization on k dimension of the matrix multiplication
        Candidates are: 'default', 'disabled', 'search'
         - 'default':
            Default parallelization strategy. A heuristic strategy is used to decide whether to parallelize on k
            dimension and the size of split factor
         - 'disabled':
            Disable parallelization on k dimension
         - 'search':
            Search for the best parallelization strategy. Takes more time but usually achieves the best performance.
        """
        self._parallel_k = strategy

    def use_fp16(self, flag=True):
        """
        Whether to use float16 data type
        """
        self._use_fp16 = flag
        return self

    def use_fp16_reduction(self, flag=True):
        """
        Whether to use float16 data type for reduction
        """
        self._use_fp16_reduction = flag
        return self

    def use_cuda_graph(self, flag=True):
        """
        Whether to use cuda graph
        """
        self._use_cuda_graph = flag
        return self

    def print_input_graph(self, flag=True):
        """
        Whether to print the input graph
        """
        self._print_input_graph = flag
        return self

    def correctness_report(self, flag=True):
        """
        Whether to check correctness and print report error
        """
        self._correctness_report = flag
        return self


dynamo_config = DynamoConfig()
