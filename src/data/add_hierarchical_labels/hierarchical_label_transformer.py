from abc import abstractmethod
from pathlib import Path


class HierarchicalLabelTransformer:
    @abstractmethod
    def add_label(self, src: Path, dst: Path) -> None:
        raise NotImplementedError
