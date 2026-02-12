import json
from typing import List, Dict, Optional
from datetime import datetime

# ---------- CONFIGURATION ----------
REDIS_TTL = 60 * 30  
MAX_STM_MESSAGES = 10  

# ---------- REDIS CONNECTION (OPTIONAL) ----------
redis_client = None
REDIS_AVAILABLE = False

try:
    import redis
    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        decode_responses=True,
        socket_connect_timeout=2,  
        socket_timeout=2
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("[STM] Redis connected successfully")
except Exception as e:
    print(f"[STM] Redis unavailable - STM disabled: {e}")
    REDIS_AVAILABLE = False
    redis_client = None


# ---------- KEY GENERATION (USER + THREAD SCOPED) ----------
def _session_key(user_id: str, thread_id: str) -> str:
    """Generate a unique key scoped to user_id AND thread_id."""
    return f"stm:{user_id}:{thread_id}"


def _context_key(user_id: str, thread_id: str) -> str:
    """Key for storing session context (last product, entity, etc.)."""
    return f"stm_ctx:{user_id}:{thread_id}"


# ---------- ADD MESSAGE TO STM ----------
async def add_stm(
    user_id: str,
    thread_id: str,
    role: str,
    content: str
) -> None:
    if not REDIS_AVAILABLE or not redis_client:
        return  
    
    if not user_id or not thread_id:
        return  
    
    try:
        key = _session_key(user_id, thread_id)
        
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        redis_client.rpush(key, message)
        redis_client.ltrim(key, -MAX_STM_MESSAGES, -1) 
        redis_client.expire(key, REDIS_TTL)
    except Exception as e:
        print(f"[STM] add_stm error (non-fatal): {e}")


# ---------- GET STM MESSAGES ----------
async def get_stm(
    user_id: str,
    thread_id: str,
    limit: int = MAX_STM_MESSAGES
) -> List[Dict]:
    if not REDIS_AVAILABLE or not redis_client:
        return [] 
    
    if not user_id or not thread_id:
        return [] 
    
    try:
        key = _session_key(user_id, thread_id)
        raw = redis_client.lrange(key, -limit, -1)
        return [json.loads(r) for r in raw] if raw else []
    except Exception as e:
        print(f"[STM] get_stm error (non-fatal): {e}")
        return []


# ---------- CLEAR STM (SESSION END) ----------
async def clear_stm(user_id: str, thread_id: str) -> None:
    if not REDIS_AVAILABLE or not redis_client:
        return
    
    if not user_id or not thread_id:
        return
    
    try:
        redis_client.delete(_session_key(user_id, thread_id))
        redis_client.delete(_context_key(user_id, thread_id))
    except Exception as e:
        print(f"[STM] clear_stm error (non-fatal): {e}")


# ---------- SESSION CONTEXT (ENTITY TRACKING) ----------
async def set_session_context(
    user_id: str,
    thread_id: str,
    context_type: str,
    value: dict
) -> None:
   
    
    if not REDIS_AVAILABLE or not redis_client:
        return
    
    if not user_id or not thread_id:
        return
    
    try:
        key = _context_key(user_id, thread_id)
        
        existing = redis_client.get(key)
        ctx = json.loads(existing) if existing else {}
        
        ctx[context_type] = value
        ctx["updated_at"] = datetime.utcnow().isoformat()
        
        redis_client.set(key, json.dumps(ctx))
        redis_client.expire(key, REDIS_TTL)
    except Exception as e:
        print(f"[STM] set_session_context error (non-fatal): {e}")


async def get_session_context(
    user_id: str,
    thread_id: str,
    context_type: Optional[str] = None
) -> Optional[dict]:
    
    
    if not REDIS_AVAILABLE or not redis_client:
        return None
    
    if not user_id or not thread_id:
        return None
    
    try:
        key = _context_key(user_id, thread_id)
        raw = redis_client.get(key)
        
        if not raw:
            return None
        
        ctx = json.loads(raw)
        
        if context_type:
            return ctx.get(context_type)
        
        return ctx
    except Exception as e:
        print(f"[STM] get_session_context error (non-fatal): {e}")
        return None


# ---------- CHECK SESSION EXISTS ----------
async def session_exists(user_id: str, thread_id: str) -> bool:
    
    if not REDIS_AVAILABLE or not redis_client:
        return False
    
    if not user_id or not thread_id:
        return False
    
    try:
        key = _session_key(user_id, thread_id)
        return redis_client.exists(key) > 0
    except Exception as e:
        print(f"[STM] session_exists error (non-fatal): {e}")
        return False


# ---------- EXTEND SESSION TTL ----------
async def touch_session(user_id: str, thread_id: str) -> None:
    
    if not REDIS_AVAILABLE or not redis_client:
        return  
    
    if not user_id or not thread_id:
        return
    
    try:
        redis_client.expire(_session_key(user_id, thread_id), REDIS_TTL)
        redis_client.expire(_context_key(user_id, thread_id), REDIS_TTL)
    except Exception as e:
        print(f"[STM] touch_session error (non-fatal): {e}")
