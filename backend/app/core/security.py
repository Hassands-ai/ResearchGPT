from datetime import datetime, timedelta
from typing import Optional
import base64
import hashlib
import hmac
import os

from jose import jwt


# ============================================================
# JWT CONFIGURATION
# ============================================================

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "paperaxiom-secret-key-change-in-production-2026"
)

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


# ============================================================
# PASSWORD HASHING
# ============================================================

PBKDF2_ITERATIONS = 310000
SALT_LENGTH = 16


def get_password_hash(password: str) -> str:
    """
    Secure password hashing using PBKDF2-HMAC-SHA256.

    This does not use bcrypt and therefore avoids the bcrypt
    72-byte password limitation and bcrypt backend problems.
    """

    if not isinstance(password, str):
        raise TypeError("Password must be a string")

    password_bytes = password.encode("utf-8")

    salt = os.urandom(SALT_LENGTH)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password_bytes,
        salt,
        PBKDF2_ITERATIONS,
    )

    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii")
    hash_b64 = base64.urlsafe_b64encode(password_hash).decode("ascii")

    return (
        f"pbkdf2_sha256${PBKDF2_ITERATIONS}"
        f"${salt_b64}${hash_b64}"
    )


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """
    Verify a PBKDF2-SHA256 password hash.
    """

    if not plain_password or not hashed_password:
        return False

    try:
        parts = hashed_password.split("$")

        if len(parts) != 4:
            return False

        algorithm = parts[0]
        iterations = int(parts[1])
        salt = base64.urlsafe_b64decode(parts[2].encode("ascii"))
        stored_hash = base64.urlsafe_b64decode(
            parts[3].encode("ascii")
        )

        if algorithm != "pbkdf2_sha256":
            return False

        calculated_hash = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt,
            iterations,
        )

        return hmac.compare_digest(
            calculated_hash,
            stored_hash,
        )

    except (ValueError, TypeError, UnicodeDecodeError):
        return False


# ============================================================
# JWT TOKEN
# ============================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
):
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

    return encoded_jwt
