from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

async def retrieve_products(
    query: str,
    k: int = 3,
    db: AsyncSession | None = None
):
    if db is None:
        raise ValueError("DB session is required")

    sql = text("""
        SELECT
            p.id,
            p.name,
            p.description,
            p.price,
            ROUND(
                    (p.total_rating / NULLIF(p.rating_count, 0))::numeric,
                    2
                ) AS avg_rating

        FROM products p
        WHERE
            p.name ILIKE :q
            OR p.description ILIKE :q
        ORDER BY p.sold_count DESC
        LIMIT :limit
    """)

    result = await db.execute(
        sql,
        {"q": f"%{query}%", "limit": k}
    )

    rows = result.mappings().all()

    return [
        {
            "product_id": r["id"],
            "name": r["name"],
            "price": r["price"],
            "rating": r["avg_rating"],
            "description": r["description"],
        }
        for r in rows
    ]


async def retrieve_reviews(
    query: str,
    k: int = 3,
    db: AsyncSession | None = None
):
    if db is None:
        raise ValueError("DB session is required")

    sql = text("""
        SELECT
            p.name AS product_name,
            fp.rating,
            f.overall_feedback,
            f.created_at
        FROM feedback_products fp
        JOIN feedbacks f ON f.feedback_id = fp.feedback_id
        JOIN products p ON p.id = fp.product_id
        WHERE
            p.name ILIKE :q
            OR f.overall_feedback ILIKE :q
        ORDER BY f.created_at DESC
        LIMIT :limit
    """)

    result = await db.execute(
        sql,
        {"q": f"%{query}%", "limit": k}
    )

    rows = result.mappings().all()

    return [
        {
            "product": r["product_name"],
            "rating": r["rating"],
            "review": r["overall_feedback"],
            "date": r["created_at"],
        }
        for r in rows
    ]
