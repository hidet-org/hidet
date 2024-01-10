# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List
from dataclasses import dataclass, asdict
import json


@dataclass
class Event:
    name: str
    cat: str
    ph: str
    ts: int
    pid: int
    tid: int
    args: Dict[str, Any]


@dataclass
class TraceEvents:
    traceEvents: List[Event]
    displayTimeUnit: str = 'ms'
    otherData: Dict[str, Any] = None


class TraceEventEmitter:
    def __init__(self, other_data: Dict[str, Any] = None):
        self.events: List[Event] = []
        self.otherData: Dict[str, Any] = other_data if other_data is not None else {}

        self.current_ts = 0

    def append(self, name: str, duration_us: int, args: Dict[str, Any] = None):
        self.events.append(
            Event(
                name=name, cat='kernel', ph='B', ts=self.current_ts, pid=0, tid=0, args=args if args is not None else {}
            )
        )
        self.current_ts += duration_us
        self.events.append(
            Event(
                name=name, cat='kernel', ph='E', ts=self.current_ts, pid=0, tid=0, args=args if args is not None else {}
            )
        )

    def export(self):
        return asdict(TraceEvents(traceEvents=self.events, otherData=self.otherData))

    def save(self, f):
        json.dump(self.export(), f)


if __name__ == '__main__':
    emitter = TraceEventEmitter()
    emitter.append('test', 1000)
    with open('test.json', 'w') as ff:
        emitter.save(ff)
