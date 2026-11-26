import smtplib
import ssl

import auth


def test_otp_email_uses_configured_smtp_transport(monkeypatch):
    calls = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls["connection"] = (host, port, timeout)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def ehlo(self):
            calls["ehlo"] = calls.get("ehlo", 0) + 1

        def starttls(self, context):
            calls["starttls"] = isinstance(context, ssl.SSLContext)

        def login(self, email, password):
            calls["login"] = (email, password)

        def send_message(self, message):
            calls["message"] = message

    monkeypatch.setenv("SMTP_EMAIL", "eventlens@example.invalid")
    monkeypatch.setenv("SMTP_PASSWORD", "test-only-password")
    monkeypatch.setenv("SMTP_HOST", "mail.example.invalid")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SMTP_FROM_NAME", "EventLens Test")
    monkeypatch.setenv("SMTP_USE_SSL", "false")
    monkeypatch.setenv("SMTP_USE_TLS", "true")
    monkeypatch.setattr(auth.smtplib, "SMTP", FakeSMTP)

    auth.send_otp_email("recipient@example.invalid", "123456", "register")

    assert calls["connection"] == ("mail.example.invalid", 2525, 20)
    assert calls["starttls"] is True
    assert calls["login"] == (
        "eventlens@example.invalid",
        "test-only-password",
    )
    assert calls["message"]["From"] == (
        "EventLens Test <eventlens@example.invalid>"
    )


def test_otp_email_rejects_non_integer_smtp_port(monkeypatch):
    monkeypatch.setenv("SMTP_EMAIL", "eventlens@example.invalid")
    monkeypatch.setenv("SMTP_PASSWORD", "test-only-password")
    monkeypatch.setenv("SMTP_PORT", "invalid")

    try:
        auth.send_otp_email("recipient@example.invalid", "123456", "register")
    except RuntimeError as error:
        assert str(error) == "SMTP_PORT must be an integer."
    else:
        raise AssertionError("Invalid SMTP port should raise an error.")


def test_otp_email_supports_implicit_tls(monkeypatch):
    calls = {}

    class FakeSMTPSSL:
        def __init__(self, host, port, timeout, context):
            calls["connection"] = (host, port, timeout)
            calls["context"] = context

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def login(self, email, password):
            calls["login"] = email

        def send_message(self, message):
            calls["message"] = message

    monkeypatch.setenv("SMTP_EMAIL", "eventlens@example.invalid")
    monkeypatch.setenv("SMTP_PASSWORD", "test-only-password")
    monkeypatch.setenv("SMTP_USE_SSL", "true")
    monkeypatch.setenv("SMTP_USE_TLS", "false")
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.setattr(auth.smtplib, "SMTP_SSL", FakeSMTPSSL)

    auth.send_otp_email("recipient@example.invalid", "123456", "registration")

    assert calls["connection"] == ("smtp.gmail.com", 465, 20)
    assert isinstance(calls["context"], ssl.SSLContext)
    assert calls["login"] == "eventlens@example.invalid"
    assert calls["message"]["To"] == "recipient@example.invalid"


def test_smtp_failure_does_not_log_credentials_or_otp(monkeypatch, caplog):
    class FailingSMTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def ehlo(self):
            pass

        def starttls(self, context):
            pass

        def login(self, *_args):
            raise smtplib.SMTPAuthenticationError(535, b"authentication failed")

    monkeypatch.setenv("SMTP_EMAIL", "private@example.invalid")
    monkeypatch.setenv("SMTP_PASSWORD", "sensitive-test-password")
    monkeypatch.setenv("SMTP_USE_TLS", "true")
    monkeypatch.setenv("SMTP_USE_SSL", "false")
    monkeypatch.setattr(auth.smtplib, "SMTP", FailingSMTP)

    try:
        auth.send_otp_email(
            "recipient@example.invalid",
            "908172",
            "registration",
        )
    except auth.EmailDeliveryError:
        pass
    else:
        raise AssertionError("SMTP authentication failure should be surfaced.")

    assert "sensitive-test-password" not in caplog.text
    assert "private@example.invalid" not in caplog.text
    assert "908172" not in caplog.text
