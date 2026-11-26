import pymysql

import database


DATABASE_ENV_KEYS = (
    "APP_ENV",
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
    "MYSQL_SSL",
    "MYSQL_SSL_CA",
    "MYSQL_SSL_VERIFY_CERT",
    "MYSQL_SSL_VERIFY_IDENTITY",
    "MYSQL_CONNECT_TIMEOUT",
    "MYSQL_READ_TIMEOUT",
    "MYSQL_WRITE_TIMEOUT",
)


def clear_database_env(monkeypatch):
    for key in DATABASE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_tidb_connection_uses_configured_tls_and_timeouts(monkeypatch):
    clear_database_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MYSQL_HOST", "tidb.example.invalid")
    monkeypatch.setenv("MYSQL_PORT", "4000")
    monkeypatch.setenv("MYSQL_USER", "eventlens")
    monkeypatch.setenv("MYSQL_PASSWORD", "test-only-password")
    monkeypatch.setenv("MYSQL_DATABASE", "eventlens")
    monkeypatch.setenv("MYSQL_SSL", "true")
    monkeypatch.setenv("MYSQL_SSL_VERIFY_CERT", "true")
    monkeypatch.setenv("MYSQL_SSL_VERIFY_IDENTITY", "true")
    monkeypatch.setenv("MYSQL_CONNECT_TIMEOUT", "7")
    monkeypatch.setenv("MYSQL_READ_TIMEOUT", "19")
    monkeypatch.setenv("MYSQL_WRITE_TIMEOUT", "23")
    captured = {}
    monkeypatch.setattr(
        database.pymysql,
        "connect",
        lambda **options: captured.update(options) or object(),
    )

    database.get_connection()

    assert captured["host"] == "tidb.example.invalid"
    assert captured["port"] == 4000
    assert captured["user"] == "eventlens"
    assert captured["password"] == "test-only-password"
    assert captured["database"] == "eventlens"
    assert captured["ssl_verify_cert"] is True
    assert captured["ssl_verify_identity"] is True
    assert captured["ssl_ca"]
    assert captured["connect_timeout"] == 7
    assert captured["read_timeout"] == 19
    assert captured["write_timeout"] == 23


def test_production_requires_tidb_host(monkeypatch):
    clear_database_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MYSQL_PORT", "4000")
    monkeypatch.setenv("MYSQL_USER", "eventlens")
    monkeypatch.setenv("MYSQL_PASSWORD", "test-only-password")
    monkeypatch.setenv("MYSQL_DATABASE", "eventlens")

    try:
        database.get_connection()
    except database.DatabaseConfigurationError as error:
        assert "MYSQL_HOST" in str(error)
        assert "test-only-password" not in str(error)
    else:
        raise AssertionError("A production database host is required.")


def test_database_connection_errors_are_sanitized(monkeypatch, caplog):
    clear_database_env(monkeypatch)
    monkeypatch.setenv("MYSQL_HOST", "tidb.example.invalid")
    monkeypatch.setenv("MYSQL_PASSWORD", "sensitive-test-password")
    monkeypatch.setattr(
        database.pymysql,
        "connect",
        lambda **_options: (_ for _ in ()).throw(
            pymysql.err.OperationalError(
                2003,
                "failed to connect with sensitive-test-password",
            )
        ),
    )

    try:
        database.get_connection()
    except database.DatabaseUnavailableError as error:
        assert str(error) == "Database service is temporarily unavailable."
    else:
        raise AssertionError("Database connection error should be surfaced.")

    assert "sensitive-test-password" not in caplog.text
    assert "tidb.example.invalid" not in caplog.text
