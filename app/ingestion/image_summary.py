import base64
import io
from app.core.model import get_vision_llm

vision_llm = get_vision_llm()

# ---------- CONVERT PIL FORMAT TO BASE64 FORMAT ----------
def pil_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---------- GET SUMMARY OF IMAGE ----------
async def summarize_image(image) -> str:
    image_b64 = pil_to_base64(image)

    response = await vision_llm.ainvoke(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image. If it is a chart, explain the trend."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    )

    return response.content
