import base64
import io
from app.core.model import get_vision_llm
from app.core.logging_config import get_ingestion_logger

# ---------- LOGGER ----------
logger = get_ingestion_logger()

vision_llm = get_vision_llm()

# ---------- CONVERT PIL FORMAT TO BASE64 FORMAT ----------
def pil_to_base64(image) -> str:
    logger.debug("[pil_to_base64] Converting image to base64")
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        result = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.debug("[pil_to_base64] Conversion successful")
        return result
    except Exception as e:
        logger.error(f"[pil_to_base64] Error converting image: {e}", exc_info=True)
        raise

# ---------- GET SUMMARY OF IMAGE ----------
async def summarize_image(image) -> str:
    logger.info("[summarize_image] Generating image summary")
    try:
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

        logger.info("[summarize_image] Summary generated successfully")
        return response.content
    except Exception as e:
        logger.error(f"[summarize_image] Error summarizing image: {e}", exc_info=True)
        raise
