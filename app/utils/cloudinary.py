from fastapi import HTTPException
from cloudinary.uploader import upload
from cloudinary.utils import cloudinary_url
from fastapi import UploadFile
MAX_IMAGE_SIZE = 5 * 1024 * 1024  


# ---------- UPLOAD IMAGE TO CLOUD ----------
async def upload_to_cloudinary(files: list[UploadFile]):
    cloud_urls = []
    
    for file in files:
        await file.seek(0)

        contents = await file.read()

        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Image '{file.filename}' exceeds 5MB limit"
            )

        await file.seek(0)
        result = upload(file.file, folder="products/")  
        cloud_urls.append(result["secure_url"])
        
    return cloud_urls

