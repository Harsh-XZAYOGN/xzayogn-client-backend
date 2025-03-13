import firebase_admin
from firebase_admin import credentials, auth
import os


cred = credentials.Certificate(os.path.abspath("app/secrets/firebase-adminsdk-prod.json"))
firebase_admin.initialize_app(cred)


def verify_auth_id_token(id_token: str) -> dict:
    return auth.verify_id_token(id_token)