from typing import List, Dict, Any
from sqlalchemy import text
from app.core.db import mongo_db
from langgraph.runtime import Runtime
from app.core.runtime import RuntimeContext

# ---------- PRODUCT DETAILS (MULTI-PRODUCT SUPPORT) ----------
async def get_product_details(
    query: str,
    runtime: Runtime[RuntimeContext],
    limit: int = 10
) -> List[Dict[str, Any]]:

    db = runtime.context.db

    sql = text("""
        SELECT
            p.id,
            p.name,
            p.description,

            -- Pricing
            p.price,
            p.selling_price,
            p.unit_type,

            -- Category
            c.name AS category_name,

            -- Ratings
            CASE
                WHEN p.rating_count > 0
                THEN ROUND((p.total_rating / NULLIF(p.rating_count, 0))::numeric, 1)
                ELSE NULL
            END AS average_rating,
            p.rating_count,

            -- Primary Image
            (
                SELECT pi.image_url
                FROM product_images pi
                WHERE pi.product_id = p.id
                AND pi.is_primary = TRUE
                LIMIT 1
            ) AS primary_image,

            -- Tags
            STRING_AGG(DISTINCT pt.name, ', ') AS tags,

            -- 🔥 Relevance Score (SAFE)
            (
                CASE WHEN p.name ILIKE '%' || $1 || '%' THEN 5 ELSE 0 END +
                CASE WHEN c.name ILIKE '%' || $1 || '%' THEN 3 ELSE 0 END +
                CASE 
                    WHEN EXISTS (
                        SELECT 1
                        FROM product_tags_map x
                        JOIN product_tags t ON t.id = x.tag_id
                        WHERE x.product_id = p.id
                        AND t.name ILIKE '%' || $1 || '%'
                    )
                    THEN 4 ELSE 0
                END +
                CASE WHEN p.description ILIKE '%' || $1 || '%' THEN 2 ELSE 0 END
            ) AS relevance_score

        FROM products p
        LEFT JOIN categories c ON c.id = p.category_id
        LEFT JOIN product_tags_map ptm ON ptm.product_id = p.id
        LEFT JOIN product_tags pt ON pt.id = ptm.tag_id

        WHERE
            p.deleted_at IS NULL
            AND p.stock > 0
            AND (
                p.name ILIKE '%' || $1 || '%'
                OR p.description ILIKE '%' || $1 || '%'
                OR c.name ILIKE '%' || $1 || '%'
                OR EXISTS (
                    SELECT 1
                    FROM product_tags_map x
                    JOIN product_tags t ON t.id = x.tag_id
                    WHERE x.product_id = p.id
                    AND t.name ILIKE '%' || $1 || '%'
                )
            )

        GROUP BY
            p.id,
            p.name,
            p.description,
            p.price,
            p.selling_price,
            p.unit_type,
            c.name,
            p.total_rating,
            p.rating_count,
            p.sold_count

        ORDER BY
            relevance_score DESC,
            p.sold_count DESC

        LIMIT $5;
    """)

    result = await db.execute(sql, {"q": query, "limit": limit})
    rows = result.mappings().all()  

    return [dict(row) for row in rows] if rows else []



# ---------- PRODUCT RATINGS ----------
async def get_product_ratings(
    query: str,
    runtime: Runtime[RuntimeContext]
):
    db = runtime.context.db

    sql = text("""
        SELECT
            ROUND(
                (total_rating / NULLIF(rating_count, 0))::numeric,
                2
                ) AS avg_rating,
            rating_count
        FROM products
        WHERE name ILIKE :q
        LIMIT 1
    """)

    result = await db.execute(sql, {"q": f"%{query}%"})
    row = result.mappings().first()

    return dict(row) if row else None


# ---------- PRODUCT REVIEWS ----------
async def get_product_reviews(
    query: str,
    runtime: Runtime[RuntimeContext],
    limit: int = 5
):
    db = runtime.context.db

    sql = text("""
        SELECT
            fp.rating,
            f.overall_feedback,
            f.created_at
        FROM feedback_products fp
        JOIN feedbacks f ON fp.feedback_id = f.feedback_id
        JOIN products p ON fp.product_id = p.id
        WHERE p.name ILIKE :q
        ORDER BY f.created_at DESC
        LIMIT :limit
    """)

    result = await db.execute(
        sql,
        {"q": f"%{query}%", "limit": limit}
    )

    rows = result.mappings().all()
    return [dict(r) for r in rows]

# ---------- TYPE OF POLICIES ----------
async def get_policy_types():
    collection = mongo_db["policy_types"]
    return list(collection.find({}, {"_id": 0}))

# ---------- POLICIES ----------
async def get_policy_by_type(query: str):
    policy_types = mongo_db["policy_types"]
    policies = mongo_db["policies"]

    policy_type = policy_types.find_one(
        {"name": {"$regex": query, "$options": "i"}}
    )

    if not policy_type:
        return None

    policy = policies.find_one(
        {"policy_type_id": policy_type["_id"]},
        {"_id": 0}
    )

    return {
        "policy_type": policy_type["name"],
        "content": policy["content"] if policy else None
    }

# ---------- FAQ ----------
async def get_faqs(is_active: bool = True):
    collection = mongo_db["faqs"]
    cursor = collection.find(
        {"is_active": is_active},
        {"_id": 0}
    )
    return list(cursor)
