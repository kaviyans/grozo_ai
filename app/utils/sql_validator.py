import re

# ---------- FORBIDDEN KEYWORDS (NON-NEGOTIABLE) ----------
FORBIDDEN_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "replace",
    "grant",
    "revoke",
    "exec",
    "execute",
    "sp_",           
    "xp_",           
    "pg_",          
    "information_schema",  
    "pg_catalog",   
}

# ---------- SCHEMA INSPECTION PATTERNS (BLOCKED) ----------
SCHEMA_INSPECTION_PATTERNS = [
    r"\binformation_schema\b",
    r"\bpg_catalog\b",
    r"\bpg_tables\b",
    r"\bpg_columns\b",
    r"\bshow\s+tables\b",
    r"\bshow\s+databases\b",
    r"\bdescribe\b",
    r"\bexplain\b",
]

SQL_COMMENT_RE = re.compile(r"(--.*?$|/\*.*?\*/)", re.MULTILINE | re.DOTALL)
SQL_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*|\s*```", re.IGNORECASE)


async def validate_sql(sql: str) -> bool:
    print("[SQL Validator] Raw SQL:", sql)

    if not sql or not sql.strip():
        raise ValueError("Empty SQL query")

    cleaned = sql.strip()
    cleaned = re.sub(SQL_CODE_FENCE_RE, "", cleaned).strip()

    cleaned = re.sub(SQL_COMMENT_RE, "", cleaned).strip()
    lowered = cleaned.lower()

    print("[SQL Validator] Cleaned SQL:", cleaned)

    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT queries are allowed")

    check_sql = cleaned.rstrip(";").strip()
    if ";" in check_sql:
        raise ValueError("Multiple SQL statements are not allowed")

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            raise ValueError(f"Forbidden SQL keyword detected: {keyword}")

    for pattern in SCHEMA_INSPECTION_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError(f"Schema inspection queries are not allowed")

    union_count = lowered.count("union")
    if union_count > 2:  
        raise ValueError("Suspicious UNION pattern detected")

    print("[SQL Validator] Validation passed")
    return True


def sanitize_sql(sql: str) -> str:
    if not sql:
        return ""
    
    cleaned = sql.strip()
    
    cleaned = re.sub(SQL_CODE_FENCE_RE, "", cleaned).strip()
    
    cleaned = re.sub(SQL_COMMENT_RE, "", cleaned).strip()
    
    cleaned = cleaned.replace("`", "").strip()
    
    return cleaned
