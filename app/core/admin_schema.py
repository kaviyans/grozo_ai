from typing import Dict, List, Literal

ADMIN_DB_SCHEMA = """
══════════════════════════════════════════════════════════════════
ADMIN DATABASE SCHEMA - E-COMMERCE PLATFORM (CANONICAL, DB-ALIGNED)
══════════════════════════════════════════════════════════════════

NOTE:
- Column names listed here EXACTLY match the database.
- Do NOT assume columns that are not listed.
- Soft delete is supported ONLY where explicitly mentioned.
- Product semantics (e.g., "healthy") must be resolved via TAGS.

────────────────────────────────────────
PRODUCT MANAGEMENT & LISTING
────────────────────────────────────────

products(
    id,
    name,
    description,
    cost_price,
    selling_price,
    price,
    stock,
    threshold,
    category_id,
    hsn_id,
    sold_count,
    total_rating,
    rating_count,
    unit_type,
    deleted_at,              -- soft delete EXISTS
    created_at,
    updated_at
)

product_images(
    id,
    product_id,
    image_url,
    is_primary
)

categories(
    id,
    name,
    upload_image,
    url_image,
    description
)

product_tags(
    id,
    name                  -- examples: healthy, organic, fitness, vegan
)

product_tags_map(
    product_id,
    tag_id
)

hsn_codes(
    id,
    hsn_code,
    description,
    gst_rate,
    effective_from,
    effective_to
)

────────────────────────────────────────
ORDER MANAGEMENT
────────────────────────────────────────

orders(
    id,
    user_id,
    total_amount,
    status,                -- USER-DEFINED ENUM
    order_type,            -- USER-DEFINED ENUM
    shipping_address,
    billing_address,
    discount_price,
    coupon_code,
    refund_amount,
    delivery_date,
    created_at,
    updated_at

    -- This table DOES NOT support soft delete
)

order_items(
    id,
    order_id,
    product_id,
    quantity,
    price,
    unit_type
)

payments(
    payment_id,
    order_id,
    payment_method,
    payment_status,
    transaction_id,
    amount,
    created_at,
    updated_at
)

refunds(
    id,
    order_id,
    user_id,
    amount,
    status,
    reason,
    created_at,
    processed_at
)

────────────────────────────────────────
USER & ACCESS CONTROL (RESTRICTED)
────────────────────────────────────────

users(
    user_id,
    name,               -- SAFE
    role_id,            -- SAFE
    created_at,
    updated_at

    -- FORBIDDEN:
    -- email, phone, password_hash, age, gender, date_of_birth
)

roles(
    role_id,
    role_name
)

role_permissions(
    role_permission_id,
    role_id,
    permission_id
)

permissions(
    permission_id,
    resource,
    create,
    read,
    update,
    delete
)

────────────────────────────────────────
CART & WISHLIST
────────────────────────────────────────

cart(
    cart_id,
    user_id,
    last_activity_at
)

cart_items(
    cart_items_id,
    cart_id,
    product_id,
    quantity,
    added_at,
    last_reminder_sent_at,
    reminder_stage
)

wishlist(
    wish_id,
    user_id
)

wishlist_items(
    wish_items_id,
    wish_id,
    product_id
)

────────────────────────────────────────
COUPONS & DISCOUNTS
────────────────────────────────────────

coupons(
    coupon_id,
    code,
    discount_type,
    discount_value,
    min_order_value,
    max_discount,
    expiry_date,
    usage_limit,
    used_count,
    per_user_limit,
    is_active,
    created_at
)

coupon_category_map(
    coupon_id,
    category_id
)

discounts(
    discount_id,
    discount_type,
    discount_value,
    product_id,
    start_date,
    end_date,
    is_active,
    created_at
)

────────────────────────────────────────
LOYALTY & POINTS
────────────────────────────────────────

loyalty_configs(
    config_id,
    loyalty_points,
    value,
    max_redeemable,
    is_active,
    created_at
)

loyalty_earn_rules(
    earn_rule_id,
    order_value,
    points,
    max_points,
    is_active,
    created_at
)

points(
    user_id,
    points
)

points_logs(
    log_id,
    user_id,
    order_id,
    points_earned,
    points_redeemed,
    created_at
)

────────────────────────────────────────
FEEDBACK & NOTIFICATIONS
────────────────────────────────────────

feedbacks(
    feedback_id,
    order_id,
    user_id,
    overall_feedback,
    created_at
)

feedback_products(
    feedback_product_id,
    feedback_id,
    product_id,
    rating
)

notifications(
    id,
    notification_code,
    type,
    title,
    message,
    user_type,
    related_order_id,
    related_product_id,
    sender_id,
    receiver_id,
    read,
    read_at,
    created_at
)

────────────────────────────────────────
SECURITY / AUTH (NEVER QUERY)
────────────────────────────────────────

sessions(
    session_id,
    user_id,
    jti,
    created_at,
    expires_at,
    revoked,
    previous_jti,
    ip_address,
    user_agent,
    last_used_at
)

revoked_tokens(
    id,
    jti,
    token_type,
    revoked_at
)

otps(
    email,
    otp,
    expires_at
)

────────────────────────────────────────
SAFE AGGREGATION RULES
────────────────────────────────────────

-- Analytics MUST be a single SELECT statement
-- Allowed aggregates ONLY:
-- COUNT(DISTINCT user_id)
-- SUM(total_amount)
-- AVG(total_rating / NULLIF(rating_count, 0))

-- NEVER SELECT PII FIELDS
══════════════════════════════════════════════════════════════════
"""



# ---------- FORBIDDEN FIELDS (PRIVACY) ----------
FORBIDDEN_FIELDS = {
    "password_hash",
    "password",
    "otp",
    "jti",
    "session_id",
    "refresh_token",
    "ip_address",
    "user_agent",
    "email",
    "phone",
    "date_of_birth",
    "age",
    "gender",
}


MASK_FIELDS = {
    "email": lambda v: f"{v[:1]}***@{v.split('@')[-1]}" if v and "@" in v else "***",
    "phone": lambda v: f"******{v[-4:]}" if v and len(v) >= 4 else "******",
}

# ---------- ADMIN INTENTS ----------
ADMIN_INTENTS = [
    "product_management",
    "product_listing",
    "order_management", 
    "user_management",
    "analytics",
    "coupon_management",
    "loyalty_management",
    "notification_management",
    "system_policy",
    "general_query"
]

# ---------- ALLOWED SQL OPERATIONS BY INTENT ----------
INTENT_SQL_PERMISSIONS = {
    "product_management": ["SELECT", "INSERT", "UPDATE"],
    "order_management": ["SELECT", "UPDATE"],
    "user_management": ["SELECT"],
    "analytics": ["SELECT"],
    "coupon_management": ["SELECT", "INSERT", "UPDATE"],
    "loyalty_management": ["SELECT", "UPDATE"],
    "notification_management": ["SELECT", "INSERT"],
    "system_policy": [], 
    "general_query": ["SELECT"],
    "product_listing": ["SELECT"]
}

# ---------- TABLES BY INTENT ----------
INTENT_TABLES = {
    "product_management": [
        "products",
        "product_images",
        "categories",
        "product_tags",
        "product_tags_map",
        "hsn_codes",
    ],
    "product_listing": [
        "products",
        "product_images",
        "categories",
        "product_tags",
        "product_tags_map",
    ],
    "order_management": [
        "orders",
        "order_items",
        "payments",
        "refunds",
        "users",
    ],
    "user_management": [
        "users",
        "roles",
        "role_permissions",
        "permissions",
    ],
    "analytics": [
        "products",
        "orders",
        "order_items",
        "payments",
        "feedbacks",
        "feedback_products",
    ],
    "coupon_management": [
        "coupons",
        "coupon_category_map",
        "discounts",
    ],
    "loyalty_management": [
        "loyalty_configs",
        "loyalty_earn_rules",
        "points",
        "points_logs",
    ],
    "notification_management": [
        "notifications",
    ],
}



PRODUCT_TAGS: Dict[str, List[str]] = {
    "hungry": ["snack", "chips", "fruit", "breakfast", "instant", "tea-time"],
    "food": ["snack", "chips", "breakfast", "fruit"],
    "tired": ["energy", "energy-boost", "caffeine", "refreshing"],
    "sleepy": ["caffeine", "tea-time", "drink"],
    "healthy": ["healthy", "organic", "natural", "low-fat", "sugar-free", "vegan"],
    "diet": ["low-fat", "sugar-free", "high-protein"],
    "quick": ["instant", "snack"],
    "drink": ["beverage", "juice", "drink", "tea-time"],
}


SuggestionType = Literal[
    "TAG_GAP",
    "TAG_COVERAGE",
    "LISTING_OPTIMIZATION",
    "CATEGORY_TAG_MISMATCH",
    "SYSTEM_IMPROVEMENT"
]
