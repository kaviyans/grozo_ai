from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

@dataclass
class RuntimeContext:
    user_id: Optional[str]
    thread_id: Optional[str]
    db: AsyncSession
