import os

from pymongo import MongoClient
from bson import ObjectId

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from datetime import datetime, timezone

from dotenv import load_dotenv


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# MONGODB CONNECTION
# ==========================================

MONGODB_URI = os.getenv(
    "MONGODB_URI"
)

if not MONGODB_URI:
    raise RuntimeError(
        "MONGODB_URI is not configured in .env"
    )


client = MongoClient(
    MONGODB_URI
)

db = client["truthlens"]


# ==========================================
# COLLECTIONS
# ==========================================

users_collection = db["users"]

predictions_collection = db["predictions"]


# ==========================================
# INDEX
# ==========================================

users_collection.create_index(
    "email",
    unique=True
)


# ==========================================
# CREATE USER
# ==========================================

def create_user(
    name,
    email,
    password
):

    name = name.strip()

    email = email.strip().lower()


    existing_user = users_collection.find_one({
        "email": email
    })


    if existing_user:

        return None


    password_hash = generate_password_hash(
        password
    )


    user = {

        "name":
        name,

        "email":
        email,

        "password_hash":
        password_hash,

        "created_at":
        datetime.now(
            timezone.utc
        )

    }


    result = users_collection.insert_one(
        user
    )


    return str(
        result.inserted_id
    )


# ==========================================
# GET USER BY EMAIL
# ==========================================

def get_user_by_email(
    email
):

    email = email.strip().lower()


    return users_collection.find_one({

        "email":
        email

    })


# ==========================================
# GET USER BY ID
# ==========================================

def get_user_by_id(
    user_id
):

    try:

        object_id = ObjectId(
            user_id
        )

    except Exception:

        return None


    return users_collection.find_one({

        "_id":
        object_id

    })


# ==========================================
# VERIFY PASSWORD
# ==========================================

def verify_password(
    password,
    password_hash
):

    return check_password_hash(

        password_hash,

        password

    )


# ==========================================
# SAVE PREDICTION
# ==========================================

def save_prediction(
    news_text,
    prediction,
    confidence,
    user_id
):

    predictions_collection.insert_one({

        "user_id":
        str(user_id),

        "news_text":
        news_text,

        "prediction":
        prediction,

        "confidence":
        confidence,

        "created_at":
        datetime.now(
            timezone.utc
        )

    })


# ==========================================
# GET USER PREDICTIONS
# ==========================================

def get_user_predictions(
    user_id
):

    return list(

        predictions_collection.find(

            {
                "user_id":
                str(user_id)
            },

            {
                "_id":
                0
            }

        ).sort(

            "created_at",

            -1

        )

    )