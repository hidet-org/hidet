class ResultEntry:
    def __init__(self) -> None:
        pass

class ResultGroup:
    def __init__(self, name: str) -> None:
        self.result_entries = []
        self.name = name
    
    def add_entry(self, entry: ResultEntry) -> None:
        self.result_entries.append(entry)
