from app.utils.jwt import encode_jwt
from app.utils.firebase import verify_auth_id_token


class UserService:

    @staticmethod
    def firebase_login(auth_token: str):
        user = verify_auth_id_token(auth_token)
        client_auth_token = encode_jwt(user)

        return {
            'user': user,
            'client_auth_token': client_auth_token,
        }
