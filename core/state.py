"""
AppContext — typed singleton that replaces the raw global STATE dict in main.py.

Responsibility:
  - Hold the runtime application state (employees, projects, model config).
  - Provide typed helper methods for common lookups.
  - Be the single source of truth for the in-process runtime data store.

Thread Safety:
  - Reads are thread-safe under the GIL.
  - Writes only happen at startup (before the server accepts requests) so no
    additional locking is required at this layer.

Determinism Level:
  High. This class holds no mutable computation state, only loaded reference data.
"""

from typing import List, Dict, Optional, Any
from core.models import Employee, ExecutionProject


class AppContext:
    """
    Typed application context. Replaces the raw global STATE dict.

    Fields are public by design — FastAPI is single-threaded at startup and
    read-only thereafter, so accessor discipline over encapsulation is correct.
    """

    def __init__(self) -> None:
        self.employees: List[Employee] = []
        self.employee_map: Dict[str, Employee] = {}
        self.projects: List[ExecutionProject] = []
        self.model_config: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def find_employee(self, employee_id: str) -> Optional[Employee]:
        """Return the Employee with the given ID, or None if not found."""
        return self.employee_map.get(employee_id)

    def has_employee(self, employee_id: Optional[str]) -> bool:
        """
        Return True if:
          - employee_id is None/empty (no targeting required), OR
          - the employee_id exists in the current employee list.
        """
        if not employee_id:
            return True
        return self.find_employee(employee_id) is not None

    def employee_count(self) -> int:
        return len(self.employees)

    def is_initialised(self) -> bool:
        """True if at least one employee was loaded at startup."""
        return bool(self.employees)


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of STATE dict.
# ---------------------------------------------------------------------------
app_ctx = AppContext()
