from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from constants import (
    MONGODB_CHUNKS_COLLECTION,
    MONGODB_DB_NAME,
    MONGODB_JOBS_COLLECTION,
    MONGODB_MESSAGES_COLLECTION,
    MONGODB_URI,
    MONGODB_UPLOADS_COLLECTION,
)


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    return MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)


def get_db() -> Database:
    return get_mongo_client()[MONGODB_DB_NAME]


def uploads_collection() -> Collection:
    return get_db()[MONGODB_UPLOADS_COLLECTION]


def chunks_collection() -> Collection:
    return get_db()[MONGODB_CHUNKS_COLLECTION]


def jobs_collection() -> Collection:
    return get_db()[MONGODB_JOBS_COLLECTION]


def messages_collection() -> Collection:
    return get_db()[MONGODB_MESSAGES_COLLECTION]
