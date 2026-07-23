"""
AWS S3 cloud storage backend (production mode).
Requires:  pip install boto3
Set env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME
"""
from typing import BinaryIO

from backend.app.core.config import get_settings
from backend.app.core.storage.base import StorageBackend

settings = get_settings()


class CloudStorage(StorageBackend):

    def __init__(self) -> None:
        try:
            import boto3  # type: ignore
            self._s3 = boto3.client(
                "s3",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )
            self._bucket = settings.S3_BUCKET_NAME
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for cloud storage. "
                "Install it with:  pip install boto3"
            ) from exc

    def save_image(self, project_id: str, filename: str, file: BinaryIO) -> str:
        key = f"{project_id}/{filename}"
        self._s3.upload_fileobj(file, self._bucket, key)
        return key

    def get_image_url(self, path: str) -> str:
        # Generate a pre-signed URL valid for 1 hour
        url = self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": path},
            ExpiresIn=3600,
        )
        return url

    def delete_image(self, path: str) -> None:
        self._s3.delete_object(Bucket=self._bucket, Key=path)

    def get_image_bytes(self, path: str) -> bytes:
        obj = self._s3.get_object(Bucket=self._bucket, Key=path)
        return obj["Body"].read()
