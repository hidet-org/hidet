from typing import Dict, Any, Optional, List, ContextManager
from contextlib import nullcontext
from time import time_ns
import json


# See also: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
# Color names: https://github.com/catapult-project/catapult/blob/main/tracing/tracing/base/color_scheme.html#L29-L72
class TraceEvent:
    def __init__(self, name, category, event_type, time_stamp, pid, tid, args: Dict[str, Any]):
        self.name = name
        self.category = category
        self.event_type = event_type
        self.time_stamp = time_stamp
        self.pid = pid
        self.tid = tid
        self.args = args if args else {}

    def export(self) -> Dict:
        event = {
            'name': self.name,
            'cat': self.category,
            'ph': self.event_type,
            'ts': self.time_stamp / 1000.0,
            'pid': self.pid,
            'tid': self.tid,
            'args': {k: str(v) for k, v in self.args.items()}
        }
        return event


class CpuTraceEvent(TraceEvent):
    def __init__(self, name, category, event_type, tid, args):
        super().__init__(name, category, event_type, time_ns(), pid=0, tid=tid, args=args)


class CudaTraceEvent(TraceEvent):
    anchor_cuda_event = None
    anchor_cuda_event_host_time = None

    def __init__(self, name, category, event_type, tid, args):
        super().__init__(name, category, event_type, None, pid=0, tid=tid, args=args)
        from hidet.runtime import cuda_event_pool
        if CudaTraceEvent.anchor_cuda_event is None:
            from hidet.ffi.cuda_api import cuda_api
            CudaTraceEvent.anchor_cuda_event = cuda_event_pool.new_event()
            cuda_api.device_synchronization()
            CudaTraceEvent.anchor_cuda_event.record_on()
            CudaTraceEvent.anchor_cuda_event_host_time = time_ns()
        self.cuda_event = cuda_event_pool.new_event()
        self.cuda_event.record_on()

    def export(self) -> Dict:
        self.time_stamp = self.cuda_event.elapsed_time_since(self.anchor_cuda_event) * 1000000.0 + self.anchor_cuda_event_host_time
        return TraceEvent.export(self)


class TraceContext:
    def __init__(self, tracer, name, category, args, trace_cuda=False):
        self.tracer: Tracer = tracer
        self.name = name
        self.category = category
        self.args = args
        self.trace_cuda = trace_cuda

    def __enter__(self):
        self.tracer.events.append(CpuTraceEvent(self.name, self.category, 'B', 0, self.args))
        if self.trace_cuda:
            self.tracer.events.append(CudaTraceEvent(self.name, self.category, 'B', 1, self.args))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace_cuda:
            self.tracer.events.append(CudaTraceEvent(self.name, self.category, 'E', 1, self.args))
        self.tracer.events.append(CpuTraceEvent(self.name, self.category, 'E', 0, self.args))


class Tracer:
    def __init__(self):
        self.events: List[TraceEvent] = []
        self.tracing: bool = False

    def export(self) -> Dict:
        from hidet.ffi.cuda_api import cuda_api
        cuda_api.device_synchronization()  # sync cuda events in trace
        ret = {
            'traceEvents': [event.export() for event in self.events],
            'displayTimeUnit': 'ns'
        }
        self.clear()
        return ret

    def dump(self, f):
        json.dump(self.export(), f)

    def clear(self):
        self.events.clear()

    def turn_on(self, turn_on=True):
        self.tracing = turn_on

    def profile(self, name: str, category: str = 'python', args: Optional[Dict[str, Any]] = None, trace_cuda=False) -> ContextManager:
        if self.tracing:
            return TraceContext(self, name, category, args, trace_cuda)
        else:
            return nullcontext()


tracer = Tracer()
