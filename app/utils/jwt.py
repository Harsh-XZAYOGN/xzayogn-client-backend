import jwt
import datetime

from app.config.settings import settings


def encode_jwt(payload: dict) -> str:
    # TODO: Setting no expiration time for now
    # payload["exp"] = datetime.datetime.now() + datetime.timedelta(seconds=3600*24*14)  # 14 days
    token = jwt.encode(payload, key=settings.JWT_SECRET)
    return token


def decode_jwt(token: str) -> dict:
    return jwt.decode(token, key=settings.JWT_SECRET, algorithms=["HS256"])


