DB_SCHEMA = """
Tables:

products(
  id, name, description, 
  price, selling_price, unit_type, stock,
  category_id, sold_count,
  total_rating, rating_count,
  deleted_at
)

categories(
  id, name
)

feedbacks(
  feedback_id, user_id, overall_feedback, created_at
)

feedback_products(
  feedback_product_id, feedback_id, product_id, rating
)

faqs(
  question, answer, is_active
)

product_images(
  id, product_id, image_url, is_primary
)

product_tags(
  id, name
)

product_tags_map(
  product_id, tag_id
)

Rules:
- Use ONLY these tables and columns
- JOIN using foreign keys
- NEVER use UPDATE, DELETE, INSERT, DROP, ALTER, TRUNCATE
- Always filter deleted_at IS NULL for products
- Use LIMIT for all queries (max 50)
"""
