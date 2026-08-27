from minio import Minio
from minio.error import S3Error
from app.core.config import settings
import io

class MinioService:
    def __init__(self):
        self.client = None
        self.bucket = settings.MINIO_BUCKET

    def _get_client(self):
        if self.client is None:
            self.client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=False
            )
            self._ensure_bucket()
        return self.client

    def _ensure_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as e:
            print(f"MinIO bucket error: {e}")

    def upload_file(self, file_data: bytes, object_name: str, content_type: str = "application/pdf") -> str:
        client = self._get_client()
        try:
            client.put_object(
                self.bucket,
                object_name,
                io.BytesIO(file_data),
                length=len(file_data),
                content_type=content_type
            )
            return f"{self.bucket}/{object_name}"
        except S3Error as e:
            raise Exception(f"Failed to upload file: {e}")

minio_service = MinioService()