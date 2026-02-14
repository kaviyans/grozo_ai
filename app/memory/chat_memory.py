from typing import List, Dict, Optional
from datetime import datetime
from app.core.db import mongo_db

# ---------- MONGODB COLLECTION ----------
messages = mongo_db["chat_memory"]

async def init_indexes():
    await messages.create_index([
        ("user_id", 1),
        ("thread_id", 1),
        ("timestamp", -1)
    ])



# ---------- SAVE MESSAGE (SESSION-SCOPED) ----------
async def save_message(user_id: str, thread_id: str, role: str, content: str, scope: str = "user") -> None:
    if not user_id or not thread_id:
        print("[chat_memory] WARNING: Missing user_id or thread_id, skipping save")
        return

    await messages.insert_one({
        "user_id": user_id,
        "thread_id": thread_id,
        "scope": scope,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    })



# ---------- GET HISTORY (SESSION-SCOPED) ----------
async def get_history(user_id: str, thread_id: str, scope: str = "user", limit: int = 10) -> List[Dict]:
    if not user_id or not thread_id:
        return []

    cursor = (
        messages.find({
            "user_id": user_id,
            "thread_id": thread_id,
            "scope": scope
        })
        .sort("timestamp", -1)
        .limit(limit)
    )

    history = await cursor.to_list(length=limit)
    history.reverse()

    return [
        {"role": msg["role"], "content": msg["content"], "timestamp": msg.get("timestamp")}
        for msg in history
    ]



# ---------- GET USER HISTORY (CROSS-THREAD, SAME USER) ----------
async def get_user_history(user_id: str, scope: str = "user", limit: int = 20) -> List[Dict]:
    if not user_id:
        return []

    cursor = messages.find({"user_id": user_id, "scope": scope}).sort("timestamp", -1).limit(limit)

    history = await cursor.to_list(length=limit)
    history.reverse()

    return [
        {
            "role": msg["role"],
            "content": msg["content"],
            "thread_id": msg.get("thread_id"),
            "timestamp": msg.get("timestamp")
        }
        for msg in history
    ]


# ---------- CLEAR THREAD HISTORY ----------
async def clear_thread_history(user_id: str, thread_id: str) -> int:
    if not user_id or not thread_id:
        return 0

    result = await messages.delete_many({"user_id": user_id, "thread_id": thread_id})
    return result.deleted_count


# ---------- CHECK THREAD HAS HISTORY ----------
async def thread_has_history(user_id: str, thread_id: str) -> bool:
    if not user_id or not thread_id:
        return False

    count = await messages.count_documents({"user_id": user_id, "thread_id": thread_id}, limit=1)
    return count > 0

