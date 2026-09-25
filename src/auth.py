import os
import jwt

from functools import wraps
from datetime import datetime, timezone, timedelta

from flask import request, jsonify

from dotenv import load_dotenv


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# JWT SECRET
# ==========================================

JWT_SECRET = os.getenv(
    "JWT_SECRET"
)


if not JWT_SECRET:

    raise RuntimeError(
        "JWT_SECRET is not configured in .env"
    )


# ==========================================
# JWT CONFIGURATION
# ==========================================

JWT_ALGORITHM = "HS256"

JWT_EXPIRATION_DAYS = 1


# ==========================================
# CREATE JWT TOKEN
# ==========================================

def create_token(user_id):

    now = datetime.now(
        timezone.utc
    )

    expiration = (
        now +
        timedelta(
            days=JWT_EXPIRATION_DAYS
        )
    )


    payload = {

        "user_id":
        str(user_id),

        "iat":
        now,

        "exp":
        expiration

    }


    token = jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )


    return token


# ==========================================
# VERIFY JWT TOKEN
# ==========================================

def verify_token(token):

    try:

        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[
                JWT_ALGORITHM
            ]
        )

        return payload


    except jwt.ExpiredSignatureError:

        return None


    except jwt.InvalidTokenError:

        return None


# ==========================================
# LOGIN REQUIRED
# ==========================================

def login_required(function):

    @wraps(function)
    def decorated_function(
        *args,
        **kwargs
    ):

        auth_header = request.headers.get(
            "Authorization"
        )


        # ==================================
        # NO AUTHORIZATION HEADER
        # ==================================

        if not auth_header:

            return jsonify({

                "error":
                "Authentication required."

            }), 401


        # ==================================
        # CHECK BEARER FORMAT
        # ==================================

        if not auth_header.startswith(
            "Bearer "
        ):

            return jsonify({

                "error":
                "Invalid authorization format."

            }), 401


        token = auth_header.split(
            " ",
            1
        )[1].strip()


        # ==================================
        # VERIFY TOKEN
        # ==================================

        payload = verify_token(
            token
        )


        if not payload:

            return jsonify({

                "error":
                "Invalid or expired token."

            }), 401


        # ==================================
        # GET USER ID
        # ==================================

        user_id = payload.get(
            "user_id"
        )


        if not user_id:

            return jsonify({

                "error":
                "Invalid token payload."

            }), 401


        # ==================================
        # ATTACH USER ID TO REQUEST
        # ==================================

        request.user_id = user_id


        return function(
            *args,
            **kwargs
        )


    return decorated_function