import logging
import os

import certifi
import pymysql
from contextlib import contextmanager
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class DatabaseUnavailableError(RuntimeError):
    """Raised when a database operation cannot be completed safely."""


class DatabaseConfigurationError(RuntimeError):
    """Raised when database environment variables are invalid."""


def _read_bool(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise DatabaseConfigurationError(f"{name} must be a boolean value.")


def _read_positive_int(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    try:
        parsed = int(value)
    except ValueError as exc:
        raise DatabaseConfigurationError(
            f"{name} must be a positive integer."
        ) from exc

    if parsed <= 0:
        raise DatabaseConfigurationError(f"{name} must be a positive integer.")
    return parsed


def get_connection():
    app_environment = os.getenv("APP_ENV", "").strip().lower()
    required_settings = (
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MYSQL_DATABASE",
    )
    if app_environment == "production":
        missing = [
            name for name in required_settings
            if not os.getenv(name, "").strip()
        ]
        if missing:
            raise DatabaseConfigurationError(
                "Required database settings are missing: "
                + ", ".join(missing)
                + "."
            )

    host = os.getenv("MYSQL_HOST", "").strip()
    if not host:
        if app_environment == "production":
            raise DatabaseConfigurationError(
                "MYSQL_HOST must be configured in production."
            )
        host = "localhost"

    is_loopback = host.lower() in {"localhost", "127.0.0.1", "::1"}
    ssl_enabled = _read_bool("MYSQL_SSL", not is_loopback)
    verify_cert = _read_bool("MYSQL_SSL_VERIFY_CERT", ssl_enabled)
    verify_identity = _read_bool("MYSQL_SSL_VERIFY_IDENTITY", verify_cert)

    if not ssl_enabled and (verify_cert or verify_identity):
        raise DatabaseConfigurationError(
            "MySQL certificate verification requires MYSQL_SSL=true."
        )
    if verify_identity and not verify_cert:
        raise DatabaseConfigurationError(
            "MYSQL_SSL_VERIFY_IDENTITY requires MYSQL_SSL_VERIFY_CERT=true."
        )
    if app_environment == "production" and not (
        ssl_enabled and verify_cert and verify_identity
    ):
        raise DatabaseConfigurationError(
            "Production database connections require certificate-verified TLS."
        )

    connection_options = {
        "host": host,
        "port": _read_positive_int(
            "MYSQL_PORT",
            4000 if app_environment == "production" else 3306,
        ),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "eventlens_db"),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
        "autocommit": False,
        "connect_timeout": _read_positive_int("MYSQL_CONNECT_TIMEOUT", 10),
        "read_timeout": _read_positive_int("MYSQL_READ_TIMEOUT", 30),
        "write_timeout": _read_positive_int("MYSQL_WRITE_TIMEOUT", 30),
    }

    if ssl_enabled:
        ssl_ca = os.getenv("MYSQL_SSL_CA") or certifi.where()
        if not os.path.isfile(ssl_ca):
            raise DatabaseConfigurationError(
                "The configured MySQL certificate authority file is unavailable."
            )
        connection_options.update({
            "ssl_ca": ssl_ca,
            "ssl_verify_cert": verify_cert,
            "ssl_verify_identity": verify_identity,
        })

    try:
        return pymysql.connect(**connection_options)
    except pymysql.MySQLError as exc:
        logging.error(
            "Database connection failed (%s).",
            type(exc).__name__,
        )
        raise DatabaseUnavailableError(
            "Database service is temporarily unavailable."
        ) from exc


def _rollback(connection):
    try:
        connection.rollback()
    except pymysql.MySQLError as exc:
        logging.warning(
            "Database rollback failed (%s).",
            type(exc).__name__,
        )


@contextmanager
def get_db_connection():
    connection = get_connection()
    try:
        yield connection
        connection.commit()
    except pymysql.err.IntegrityError:
        _rollback(connection)
        raise
    except pymysql.MySQLError as exc:
        _rollback(connection)
        logging.error(
            "Database operation failed (%s).",
            type(exc).__name__,
        )
        raise DatabaseUnavailableError(
            "Database service is temporarily unavailable."
        ) from exc
    except Exception:
        _rollback(connection)
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
                CREATE TABLE IF NOT EXISTS pending_registrations (
                    email VARCHAR(255) PRIMARY KEY,
                    full_name VARCHAR(150) NOT NULL,
                    otp_hash VARCHAR(255) NOT NULL,
                    expires_at DATETIME NOT NULL,
                    attempts INT NOT NULL DEFAULT 0,
                    verified_at DATETIME NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_pending_registration_expiry (expires_at)
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


def verify_required_tables():
    required = {
        "users",
        "email_otps",
        "pending_registrations",
        "sessions",
        "reports",
    }
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.execute("SHOW TABLES")
            rows = cursor.fetchall()

    existing = {next(iter(row.values())) for row in rows}
    missing = required - existing
    if missing:
        raise DatabaseUnavailableError(
            "Database schema is incomplete. Required application tables are missing."
        )
    return existing


if __name__ == "__main__":
    initialize_database()
    print("EventLens MySQL database initialized successfully.")
