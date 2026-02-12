import uuid

# ---------- GENERATE THREAD ID ----------
async def generate_thread_id() -> str:
    return f"thread-{uuid.uuid4().hex}"
