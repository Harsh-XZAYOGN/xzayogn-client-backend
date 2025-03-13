from fastapi import APIRouter
from app.services.user_service import UserService

router = APIRouter(
    tags=["no_auth"],
    responses={404: {"description": "Not found"}},
)


@router.post("/user/firebase-login")
async def login(auth_token: str):
    return UserService.firebase_login(auth_token=auth_token)

