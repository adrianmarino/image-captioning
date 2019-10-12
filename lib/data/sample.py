from lib.utils import plot_utils


class Sample:
    def __init__(self, id, path):
        self.id = id
        self.path = path
        self.descriptions = []

    def add_desc(self, value): self.descriptions.append(value)

    def as_json(self): return plot_utils.as_json(self)
