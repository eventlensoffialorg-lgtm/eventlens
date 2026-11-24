
import os
import pymysql
from contextlib import contextmanager
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))


def get_connection():
    connection = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "eventlens_db"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )

    return connection


@contextmanager
def get_db_connection():
    connection = get_connection()

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def initialize_database():

    with get_db_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id CHAR(36) PRIMARY KEY,
                    full_name VARCHAR(150) NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    email_verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_otps (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    otp_hash VARCHAR(255) NOT NULL,
                    purpose VARCHAR(30) NOT NULL,
                    expires_at DATETIME NOT NULL,
                    attempts INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_otp_email (email)
                ) ENGINE=InnoDB
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id CHAR(36) NOT NULL,
                    token_hash CHAR(64) NOT NULL UNIQUE,
                    expires_at DATETIME NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
                ) ENGINE=InnoDB
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id VARCHAR(100) PRIMARY KEY,
                    user_id CHAR(36) NOT NULL,
                    created_at VARCHAR(50) NOT NULL,
                    report_json LONGTEXT NOT NULL,
                    INDEX idx_reports_user_created (user_id, created_at),
                    FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
                ) ENGINE=InnoDB
            """)


if __name__ == "__main__":

    initialize_database()

    print("EventLens MySQL database initialized successfully.")