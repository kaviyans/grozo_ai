from openai import OpenAI
import os
import base64
from app.core.model import get_vision_llm

vision_llm = get_vision_llm()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# ---------- ANALYZE IMAGE CONTENT ----------
async def analyze_product_image(image_url: str) -> str:
    response = client.responses.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this product image"},
                    {
                        "type": "input_image",
                        "image_url": image_url
                    }
                ]
            }
        ]
    )
    return response.output_text

