from sqlalchemy import text
from typing import Dict, Any, List, Optional
from app.core.db import engine

async def execute_sql(
    sql: str,
    params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    async with engine.connect() as conn:
        result = await conn.execute(
            text(sql),
            params or {}
        )
        rows = result.mappings().all()
        return [dict(row) for row in rows]
