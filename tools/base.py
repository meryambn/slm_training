from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": round(self.execution_time, 4),
            "metadata": self.metadata
        }

class BaseTool(ABC):
    name: str
    description: str
    parameters_schema: Dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with keyword arguments and return a ToolResult."""
        pass
