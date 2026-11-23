from __future__ import annotations

from abc import ABC, abstractmethod
from boardforge.models import Job, SourceRef


class ConnectorError(RuntimeError):
    pass


class JobConnector(ABC):
    kind: str

    @abstractmethod
    async def fetch(self, source: SourceRef) -> list[Job]:
        raise NotImplementedError
