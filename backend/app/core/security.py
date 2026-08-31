from datetime import datetime, timedelta
from typing import Optional
import hashlib

from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings


# ============================================================
# PASSWORD HASHING
# ============================================================
#
# bcrypt has a 72-byte password limitation.
# We SHA-256 pre-hash passwords before sending them to bcrypt.
#
# This allows passwords of any reasonable length while still
# using bcrypt for the final password hash.
#
# Verification also supports OLD bcrypt hashes created before
# this fix, so existing users are not broken.
# ============================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


def _prepare_password(password: str) -> str:
    """
    Convert the password into a fixed-length SHA-256 hex digest
    before bcrypt processing.

    SHA-256 hex digest = 64 ASCII characters,
    safely below bcrypt's 72-byte limit.
    """
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """
    Verify a password.

    First checks the new SHA-256 + bcrypt format.
    Then falls back to the old direct-bcrypt format so
    previously registered users continue to work.
    """

    # New password format
    try:
        if pwd_context.verify(
            _prepare_password(plain_password),
            hashed_password,
        ):
            return True
    except Exception:
        pass

    # Backward compatibility for existing users
    # whose passwords were hashed directly with bcrypt.
    try:
        if len(plain_password.encode("utf-8")) <= 72:
            return pwd_context.verify(
                plain_password,
                hashed_password,
            )
    except Exception:
        pass

    return False


def get_password_hash(password: str) -> str:
    """
    Create a bcrypt hash from the SHA-256-prepared password.
    """
    return pwd_context.hash(
        _prepare_password(password)
    )


# ============================================================
# JWT CONFIGURATION
# ============================================================

SECRET_KEY = "paperaxiom-secret-key-change-in-production-2026"
ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# ============================================================
# ACCESS TOKEN
# ============================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
):
    """
    Create a JWT access token.
    """

    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update(
        {
            "exp": expire,
        }
    )

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

    return encoded_jwt
