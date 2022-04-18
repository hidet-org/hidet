from typing import Dict, Any, Optional, List, ContextManager
from contextlib import nullcontext
from time import time_ns
import json


# See also: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
# Color names: https://github.com/catapult-project/catapult/blob/main/tracing/tracing/base/color_scheme.html#L29-L72
class TraceEvent:
    def __init__(self, name, category, event_type, time_stamp, pid: int = 0, tid: int = 0, args: Optional[Dict[str, Any]] = None, color_name: Optional[str] = None):
        self.name = name
        self.category = category
        self.event_type = event_type
        self.time_stamp = time_stamp
        self.pid = pid
        self.tid = tid
        self.args = args if args else {}
        self.color_name = color_name if color_name else None

    def export(self) -> Dict:
        event = {
            'name': self.name,
            'cat': self.category,
            'ph': self.event_type,
            'ts': self.time_stamp / 1000000.0,
            'pid': self.pid,
            'tid': self.tid,
        }
        if self.args:
            event['args'] = {k: str(v) for k, v in self.args.items()}
        if self.color_name:
            event['cname'] = self.color_name
        return event


class CudaTraceEvent(TraceEvent):
    start_cuda_event = None
    start_cuda_event_cuda_time = 0.0
    start_cuda_event_host_time = None

    def __init__(self, name, category, event_type, time_stamp):
        super().__init__(name, category, event_type, time_stamp)
        self.cuda_event = None


class TraceContext:
    def __init__(self, tracer, name, category, args):
        self.tracer: Tracer = tracer
        self.name = name
        self.category = category
        self.args = args

    def __enter__(self):
        self.tracer.append(self.name, self.category, 'B', time_ns(), args=self.args)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracer.append(self.name, self.category, 'E', time_ns())


class Tracer:
    def __init__(self):
        self.events: List[TraceEvent] = []
        self.tracing: bool = False

    def append(self, name, category, event_type, time_stamp, pid: int = 0, tid: int = 0, args: Optional[Dict[str, Any]] = None, color_name: Optional[str] = None):
        self.events.append(TraceEvent(name, category, event_type, time_stamp, pid, tid, args, color_name))

    def export(self) -> Dict:
        return {
            'traceEvents': [event.export() for event in self.events],
            'displayTimeUnit': 'ns'
        }

    def dump(self, f):
        json.dump(self.export(), f)
        self.clear()

    def clear(self):
        self.events.clear()

    def turn_on(self, turn_on=True):
        self.tracing = turn_on

    def profile(self, name: str, category: str = 'python', args: Optional[Dict[str, Any]] = None) -> ContextManager:
        if self.tracing:
            return TraceContext(self, name, category, args)
        else:
            return nullcontext()

    def profile_cuda(self, name: str, category: str = 'cuda', args: Optional[Dict[str, Any]] = None) -> ContextManager:
        raise NotImplementedError()


tracer = Tracer()
