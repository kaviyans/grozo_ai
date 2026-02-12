import re
from typing import Optional, Tuple, List
from app.core.admin_schema import (
    FORBIDDEN_FIELDS,
    MASK_FIELDS,
    INTENT_SQL_PERMISSIONS,
    INTENT_TABLES
)

# ---------- FORBIDDEN KEYWORDS ----------
ADMIN_FORBIDDEN_KEYWORDS = {
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
    "exec",
    "execute",
    "information_schema",
    "pg_catalog",
    "pg_tables",
}

# ---------- REGEX PATTERNS ----------
SQL_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*|\s*```", re.IGNORECASE)
SQL_COMMENT_RE = re.compile(r"(--.*?$|/\*.*?\*/)", re.MULTILINE | re.DOTALL)


def sanitize_admin_sql(sql: str) -> str:
    if not sql:
        return ""
    
    cleaned = sql.strip()
    cleaned = re.sub(SQL_CODE_FENCE_RE, "", cleaned).strip()
    cleaned = re.sub(SQL_COMMENT_RE, "", cleaned).strip()
    cleaned = cleaned.replace("`", "").strip()
    
    return cleaned


def detect_sql_operation(sql: str) -> str:
    lowered = sql.strip().lower()
    
    if lowered.startswith("select") or lowered.startswith("with"):
        return "SELECT"
    elif lowered.startswith("insert"):
        return "INSERT"
    elif lowered.startswith("update"):
        return "UPDATE"
    elif lowered.startswith("delete"):
        return "DELETE"
    
    return "UNKNOWN"


def check_privacy_violation(sql: str) -> Tuple[bool, Optional[str]]:
    lowered = sql.lower()
    
    for field in FORBIDDEN_FIELDS:
        patterns = [
            rf"\b{field}\b",  
            rf"\.{field}\b",  
            rf"\"{field}\"",  
        ]
        
        for pattern in patterns:
            if re.search(pattern, lowered):
                return True, field
    
    return False, None


def check_intent_permission(
    sql: str,
    intent: str,
    operation: str
) -> Tuple[bool, Optional[str]]:
    allowed_ops = INTENT_SQL_PERMISSIONS.get(intent, [])
    
    if operation not in allowed_ops:
        return False, f"Operation '{operation}' not allowed for intent '{intent}'"
    
    return True, None


def check_table_access(sql: str, intent: str) -> Tuple[bool, Optional[str]]:
    allowed_tables = INTENT_TABLES.get(intent, [])
    
    if not allowed_tables:
        return True, None 
    
    lowered = sql.lower()
    
    table_patterns = [
        r"\bfrom\s+(\w+)",
        r"\bjoin\s+(\w+)",
        r"\binto\s+(\w+)",
        r"\bupdate\s+(\w+)",
    ]
    
    accessed_tables = set()
    for pattern in table_patterns:
        matches = re.findall(pattern, lowered)
        accessed_tables.update(matches)
    
    for table in accessed_tables:
        if table not in allowed_tables and table not in ["as", "on", "where"]:
            if intent == "analytics" and table == "users":
                continue  
            
            return False, f"Table '{table}' not accessible for intent '{intent}'"
    
    return True, None


async def validate_admin_sql(
    sql: str,
    intent: str
) -> Tuple[bool, Optional[str]]:
    if not sql or not sql.strip():
        return False, "Empty SQL query"

    print("INTENT:", intent)
    print("PERMS:", INTENT_SQL_PERMISSIONS.get(intent))
    print("TABLES:", INTENT_TABLES.get(intent))

    
    cleaned = sanitize_admin_sql(sql)
    lowered = cleaned.lower()
    
    print(f"[Admin SQL Validator] Validating: {cleaned[:100]}...")
    
    is_violation, violated_field = check_privacy_violation(cleaned)
    
    if is_violation:
        return False, f"Access to protected field '{violated_field}' is forbidden"
    
    for keyword in ADMIN_FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            return False, f"Forbidden SQL keyword: {keyword}"
    
    check_sql = cleaned.rstrip(";").strip()
    if ";" in check_sql:
        return False, "Multiple SQL statements not allowed"
    
    operation = detect_sql_operation(cleaned)
    if operation == "UNKNOWN":
        return False, "Unknown SQL operation type"
    
    is_allowed, error = check_intent_permission(cleaned, intent, operation)
    if not is_allowed:
        return False, error
    
    is_allowed, error = check_table_access(cleaned, intent)
    if not is_allowed:
        return False, error
    
    if operation == "DELETE":
        return False, "DELETE operations require explicit admin confirmation"
    
    print("[Admin SQL Validator] Validation passed")
    return True, None


def mask_sensitive_data(data: dict) -> dict:
    if not data:
        return data
    
    masked = dict(data)
    
    for field, mask_func in MASK_FIELDS.items():
        if field in masked and masked[field]:
            masked[field] = mask_func(str(masked[field]))
    
    return masked


def mask_result_rows(rows: List[dict]) -> List[dict]:
    """Apply masking to a list of result rows."""
    return [mask_sensitive_data(row) for row in rows]
