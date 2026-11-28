from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib

from flask import Flask
from werkzeug.security import check_password_hash, generate_password_hash

import auth


class FakeCursor:
    def __init__(self, fetch_results=()):
        self.fetch_results = list(fetch_results)
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, parameters=None):
        self.executed.append((" ".join(query.lower().split()), parameters))

    def fetchone(self):
        if self.fetch_results:
            return self.fetch_results.pop(0)
        return None


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class RegistrationCompletionCursor(FakeCursor):
    def __init__(self, pending):
        super().__init__()
        self.pending = pending
        self.user = None
        self._result = None
        self.session = None

    def execute(self, query, parameters=None):
        normalized = " ".join(query.lower().split())
        super().execute(query, parameters)

        if normalized.startswith("select full_name, expires_at, verified_at"):
            self._result = self.pending
        elif normalized.startswith("select id, email_verified from users"):
            self._result = None
        elif normalized.startswith("insert into users"):
            self.user = {
                "id": parameters[0],
                "full_name": parameters[1],
                "email": parameters[2],
                "password_hash": parameters[3],
            }
        elif normalized.startswith("select id, full_name, email from users"):
            self._result = self.user
        elif normalized.startswith("insert into sessions"):
            self.session = parameters
        elif normalized.startswith("delete from pending_registrations"):
            self.pending = None

    def fetchone(self):
        result = self._result
        self._result = None
        return result


def install_fake_database(monkeypatch, cursor):
    connection = FakeConnection(cursor)

    @contextmanager
    def get_db_connection():
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()

    monkeypatch.setattr(auth, "get_db_connection", get_db_connection)
    return connection


def make_client():
    app = Flask(__name__)
    app.register_blueprint(auth.auth_bp, url_prefix="/api/auth")
    return app.test_client()


def test_registration_persists_only_hashed_otp_before_verification(monkeypatch):
    cursor = FakeCursor([None, None])
    connection = install_fake_database(monkeypatch, cursor)
    sent = {}
    monkeypatch.setattr(
        auth,
        "send_otp_email",
        lambda email, otp, _purpose: sent.update(email=email, otp=otp),
    )

    auth.create_registration_otp("person@example.com", "Example Person")

    insert = next(
        statement for statement in cursor.executed
        if statement[0].startswith("insert into pending_registrations")
    )
    assert insert[1][0:2] == ("person@example.com", "Example Person")
    assert check_password_hash(insert[1][2], sent["otp"])
    assert sent["email"] == "person@example.com"
    assert connection.commits == 1


def test_registration_rejects_an_existing_verified_account(monkeypatch):
    cursor = FakeCursor([{"id": "existing-user", "email_verified": True}])
    install_fake_database(monkeypatch, cursor)
    monkeypatch.setattr(
        auth,
        "send_otp_email",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("OTP email must not be sent to an existing account.")
        ),
    )

    try:
        auth.create_registration_otp("person@example.com", "Example Person")
    except auth.DuplicateAccountError:
        pass
    else:
        raise AssertionError("Verified duplicate registration should be rejected.")


def test_resending_registration_otp_supersedes_previous_hash(monkeypatch):
    now = datetime.now()
    old_code = "654321"
    cursor = FakeCursor([
        None,
        {"created_at": now - timedelta(seconds=auth.OTP_RESEND_SECONDS + 1)},
    ])
    install_fake_database(monkeypatch, cursor)
    sent = {}
    monkeypatch.setattr(auth.secrets, "randbelow", lambda _limit: 123456)
    monkeypatch.setattr(
        auth,
        "send_otp_email",
        lambda email, otp, _purpose: sent.update(email=email, otp=otp),
    )
    previous_hash = generate_password_hash(old_code)
    cursor.fetch_results[1] = {
        "created_at": now - timedelta(seconds=auth.OTP_RESEND_SECONDS + 1)
    }

    auth.create_registration_otp("person@example.com", "Example Person")

    update = next(
        statement for statement in cursor.executed
        if statement[0].startswith("update pending_registrations")
    )
    replacement_hash = update[1][1]
    assert replacement_hash != previous_hash
    assert check_password_hash(replacement_hash, sent["otp"])
    assert not check_password_hash(replacement_hash, old_code)
    assert "attempts = 0" in update[0]
    assert "verified_at = null" in update[0]


def test_registration_otp_verification_marks_code_used(monkeypatch):
    otp = "123456"
    cursor = FakeCursor([{
        "otp_hash": generate_password_hash(otp),
        "expires_at": datetime.now() + timedelta(minutes=2),
        "attempts": 0,
        "verified_at": None,
    }])
    install_fake_database(monkeypatch, cursor)

    valid, message = auth.verify_registration_otp("person@example.com", otp)

    assert valid is True
    assert message == "OTP verified."
    assert any(
        statement[0].startswith("update pending_registrations")
        and statement[1][1] == "person@example.com"
        for statement in cursor.executed
    )


def test_registration_otp_rejects_expired_and_already_used_codes(monkeypatch):
    expired_cursor = FakeCursor([{
        "otp_hash": generate_password_hash("123456"),
        "expires_at": datetime.now() - timedelta(seconds=1),
        "attempts": 0,
        "verified_at": None,
    }])
    install_fake_database(monkeypatch, expired_cursor)

    valid, message = auth.verify_registration_otp(
        "person@example.com",
        "123456",
    )

    assert valid is False
    assert message == "OTP has expired."
    assert any(
        statement[0].startswith("delete from pending_registrations")
        for statement in expired_cursor.executed
    )

    used_cursor = FakeCursor([{
        "otp_hash": generate_password_hash("123456"),
        "expires_at": datetime.now() + timedelta(minutes=2),
        "attempts": 0,
        "verified_at": datetime.now(),
    }])
    install_fake_database(monkeypatch, used_cursor)

    valid, message = auth.verify_registration_otp(
        "person@example.com",
        "123456",
    )

    assert valid is False
    assert message == "OTP has already been used."


def test_registration_otp_rejects_incorrect_and_excessive_attempts(monkeypatch):
    cursor = FakeCursor([{
        "otp_hash": generate_password_hash("123456"),
        "expires_at": datetime.now() + timedelta(minutes=2),
        "attempts": 0,
        "verified_at": None,
    }])
    install_fake_database(monkeypatch, cursor)

    valid, message = auth.verify_registration_otp(
        "person@example.com",
        "000000",
    )

    assert valid is False
    assert message == "Incorrect OTP."
    assert any(
        statement[0].startswith("update pending_registrations")
        and statement[1][0] == 1
        for statement in cursor.executed
    )

    limited_cursor = FakeCursor([{
        "otp_hash": generate_password_hash("123456"),
        "expires_at": datetime.now() + timedelta(minutes=2),
        "attempts": auth.OTP_MAX_ATTEMPTS - 1,
        "verified_at": None,
    }])
    install_fake_database(monkeypatch, limited_cursor)

    valid, message = auth.verify_registration_otp(
        "person@example.com",
        "000000",
    )

    assert valid is False
    assert "Maximum OTP attempts" in message
    assert any(
        statement[0].startswith("delete from pending_registrations")
        for statement in limited_cursor.executed
    )


def test_registration_requires_otp_before_accepting_password(monkeypatch):
    client = make_client()
    monkeypatch.setattr(
        auth,
        "create_registration_otp",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("OTP should not be requested when a password is sent.")
        ),
    )

    response = client.post(
        "/api/auth/register",
        json={
            "fullName": "Example Person",
            "email": "person@example.com",
            "password": "unverified-password",
        },
    )

    assert response.status_code == 400
    assert "Verify your email" in response.get_json()["error"]


def test_registration_completion_does_not_hash_password_before_verification(
    monkeypatch,
):
    cursor = FakeCursor([{
        "full_name": "Example Person",
        "expires_at": datetime.now() + timedelta(minutes=2),
        "verified_at": None,
    }])
    connection = install_fake_database(monkeypatch, cursor)
    client = make_client()
    monkeypatch.setattr(
        auth,
        "generate_password_hash",
        lambda _password: (_ for _ in ()).throw(
            AssertionError("Password must not be hashed before OTP verification.")
        ),
    )

    response = client.post(
        "/api/auth/complete-registration",
        json={
            "email": "person@example.com",
            "password": "secure-password",
        },
    )

    assert response.status_code == 403
    assert "Verify your email" in response.get_json()["error"]
    assert connection.commits == 1
    assert not any(
        statement[0].startswith("insert into users")
        for statement in cursor.executed
    )


def test_registration_completion_saves_password_and_session_atomically(
    monkeypatch,
):
    cursor = RegistrationCompletionCursor({
        "full_name": "Example Person",
        "expires_at": datetime.now() + timedelta(minutes=2),
        "verified_at": datetime.now(),
    })
    connection = install_fake_database(monkeypatch, cursor)
    client = make_client()

    response = client.post(
        "/api/auth/complete-registration",
        json={
            "email": "person@example.com",
            "password": "secure-password",
        },
    )

    result = response.get_json()
    assert response.status_code == 200
    assert result["user"]["email"] == "person@example.com"
    assert check_password_hash(
        cursor.user["password_hash"],
        "secure-password",
    )
    assert cursor.session[0] == cursor.user["id"]
    assert cursor.session[1] == hashlib.sha256(
        result["token"].encode()
    ).hexdigest()
    assert cursor.pending is None
    assert connection.commits == 1


def test_login_accepts_valid_credentials_and_rejects_invalid_password(monkeypatch):
    stored_hash = generate_password_hash("valid-password")
    monkeypatch.setattr(
        auth,
        "get_user",
        lambda _email: {
            "id": "existing-user",
            "full_name": "Example Person",
            "email": "person@example.com",
            "password_hash": stored_hash,
            "email_verified": True,
        },
    )
    monkeypatch.setattr(auth, "create_session", lambda _user_id: "session-token")
    client = make_client()

    valid_response = client.post(
        "/api/auth/login",
        json={
            "email": "person@example.com",
            "password": "valid-password",
        },
    )
    invalid_response = client.post(
        "/api/auth/login",
        json={
            "email": "person@example.com",
            "password": "incorrect-password",
        },
    )

    assert valid_response.status_code == 200
    assert valid_response.get_json()["token"] == "session-token"
    assert invalid_response.status_code == 401
