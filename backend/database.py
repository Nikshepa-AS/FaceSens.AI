import sqlite3
from datetime import datetime


DATABASE_FILE = "data/facesense.db"


def get_connection():

    return sqlite3.connect(
        DATABASE_FILE
    )


def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()


    # --------------------------------
    # Users table
    # --------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL UNIQUE,

            created_at TEXT NOT NULL

        )
    """)


    # --------------------------------
    # Recognition history
    # --------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT,

            similarity REAL NOT NULL,

            status TEXT NOT NULL,

            timestamp TEXT NOT NULL

        )
    """)


    connection.commit()

    connection.close()


def add_user(name):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO users
        (name, created_at)
        VALUES (?, ?)
        """,
        (
            name,
            datetime.now().isoformat()
        )
    )

    connection.commit()

    connection.close()


def log_recognition(
    name,
    similarity,
    status
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO recognition_history
        (name, similarity, status, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (
            name,
            similarity,
            status,
            datetime.now().isoformat()
        )
    )

    connection.commit()

    connection.close()
def delete_user(name):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM users
            WHERE name = ?
            """,
            (name,)
        )

        connection.commit()

        return cursor.rowcount > 0

    finally:
        connection.close()    
    