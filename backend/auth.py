import os
import uuid
import hashlib
import secrets
import smtplib

from datetime import datetime, timedelta
from email.message import EmailMessage
from functools import wraps

from flask import Blueprint, request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db_connection
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

auth_bp = Blueprint("auth", __name__)

OTP_EXPIRY_MINUTES = 5
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_SECONDS = 60
SESSION_DAYS = 7


def send_otp_email(email, otp, purpose):
    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_email or not smtp_password:
        raise RuntimeError("Email service is not configured.")

    message = EmailMessage()
    message["Subject"] = "Your EventLens verification code"
    message["From"] = f"EventLens <{smtp_email}>"
    message["To"] = email

    message.set_content(
        f"""
Hello,

Your EventLens verification code is:

{otp}

This code expires in 5 minutes.

If you did not request this code, please ignore this email.

EventLens Team
"""
    )

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_email, smtp_password)
        server.send_message(message)


def create_otp(email, purpose):
    now = datetime.now()

    with get_db_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT created_at
                FROM email_otps
                WHERE email = %s AND purpose = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (email, purpose),
            )

            previous = cursor.fetchone()

            if previous:
                elapsed = (now - previous["created_at"]).total_seconds()

                if elapsed < OTP_RESEND_SECONDS:
                    raise ValueError("Please wait before requesting another OTP.")

            otp = f"{secrets.randbelow(1000000):06d}"

            cursor.execute(
                """
                DELETE FROM email_otps
                WHERE email = %s AND purpose = %s
                """,
                (email, purpose),
            )

            cursor.execute(
                """
                INSERT INTO email_otps
                (email, otp_hash, purpose, expires_at, attempts)
                VALUES (%s, %s, %s, %s, 0)
                """,
                (
                    email,
                    generate_password_hash(otp),
                    purpose,
                    now + timedelta(minutes=OTP_EXPIRY_MINUTES),
                ),
            )

    try:
        send_otp_email(email, otp, purpose)
    except Exception:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM email_otps WHERE email = %s AND purpose = %s",
                    (email, purpose),
                )
        raise

    return True


def verify_otp(email, otp, purpose):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT *
                FROM email_otps
                WHERE email = %s AND purpose = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (email, purpose),
            )

            record = cursor.fetchone()

            if not record:
                return False, "OTP not found. Request a new OTP."

            if record["expires_at"] <= datetime.now():
                cursor.execute(
                    "DELETE FROM email_otps WHERE id = %s",
                    (record["id"],),
                )
                return False, "OTP has expired."

            if record["attempts"] >= OTP_MAX_ATTEMPTS:
                cursor.execute(
                    "DELETE FROM email_otps WHERE id = %s",
                    (record["id"],),
                )
                return False, "Maximum OTP attempts exceeded."

            if not check_password_hash(record["otp_hash"], otp):
                cursor.execute(
                    """
                    UPDATE email_otps
                    SET attempts = attempts + 1
                    WHERE id = %s
                    """,
                    (record["id"],),
                )
                return False, "Incorrect OTP."

            cursor.execute(
                "DELETE FROM email_otps WHERE id = %s",
                (record["id"],),
            )

    return True, "OTP verified."


def create_session(user_id):
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO sessions (user_id, token_hash, expires_at)
                VALUES (%s, %s, %s)
                """,
                (
                    user_id,
                    token_hash,
                    datetime.now() + timedelta(days=SESSION_DAYS),
                ),
            )

    return token


def get_user(email):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, full_name, email, password_hash, email_verified
                FROM users
                WHERE email = %s
                """,
                (email,),
            )
            return cursor.fetchone()


def user_response(user):
    return {
        "id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
    }


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}

    full_name = str(data.get("fullName", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not full_name or not email or not password:
        return jsonify({"error": "All fields are required."}), 400

    if len(full_name) > 150 or len(email) > 255:
        return jsonify({"error": "Input is too long."}), 400

    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"error": "Enter a valid email address."}), 400

    if len(password) < 8:
        return jsonify({"error": "Password must contain at least 8 characters."}), 400

    existing = get_user(email)

    if existing and existing["email_verified"]:
        return jsonify({"error": "An account with this email already exists."}), 409

    user_id = existing["id"] if existing else str(uuid.uuid4())

    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                if existing:
                    cursor.execute(
                        """
                        UPDATE users
                        SET full_name = %s, password_hash = %s
                        WHERE id = %s
                        """,
                        (full_name, generate_password_hash(password), user_id),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO users
                        (id, full_name, email, password_hash, email_verified)
                        VALUES (%s, %s, %s, %s, FALSE)
                        """,
                        (
                            user_id,
                            full_name,
                            email,
                            generate_password_hash(password),
                        ),
                    )

        create_otp(email, "registration")

        return jsonify({
            "message": "Verification OTP sent to your email.",
            "email": email,
            "verification_required": True,
        }), 200

    except ValueError as error:
        return jsonify({"error": str(error)}), 429
    except Exception:
        return jsonify({"error": "Unable to start registration. Check email service configuration."}), 503


@auth_bp.route("/verify-otp", methods=["POST"])
def verify_registration():
    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()
    otp = str(data.get("otp", "")).strip()

    if not email or len(otp) != 6 or not otp.isdigit():
        return jsonify({"error": "Enter a valid six-digit OTP."}), 400

    valid, message = verify_otp(email, otp, "registration")

    if not valid:
        return jsonify({"error": message}), 400

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET email_verified = TRUE
                WHERE email = %s
                """,
                (email,),
            )

            cursor.execute(
                """
                SELECT id, full_name, email
                FROM users
                WHERE email = %s AND email_verified = TRUE
                """,
                (email,),
            )

            user = cursor.fetchone()

    if not user:
        return jsonify({"error": "Account verification failed."}), 400

    token = create_session(user["id"])

    return jsonify({
        "message": "Account verified successfully.",
        "token": token,
        "user": user_response(user),
    })


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    user = get_user(email)

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password."}), 401

    if not user["email_verified"]:
        return jsonify({"error": "Please verify your email first."}), 403

    token = create_session(user["id"])

    return jsonify({
        "message": "Login successful.",
        "token": token,
        "user": user_response(user),
    })


@auth_bp.route("/logout", methods=["POST"])
def logout():
    authorization = request.headers.get("Authorization", "")

    if not authorization.startswith("Bearer "):
        return jsonify({"message": "Logged out."})

    token = authorization.split(" ", 1)[1].strip()
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM sessions WHERE token_hash = %s",
                (token_hash,),
            )

    return jsonify({"message": "Logged out successfully."})


@auth_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()

    if not email:
        return jsonify({"error": "Email is required."}), 400

    user = get_user(email)

    if user and user["email_verified"]:
        try:
            create_otp(email, "password_reset")
        except ValueError as error:
            return jsonify({"error": str(error)}), 429
        except Exception:
            return jsonify({"error": "Unable to send verification email."}), 503

    return jsonify({
        "message": "If the account exists, a password reset OTP has been sent."
    })


@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()
    otp = str(data.get("otp", "")).strip()
    new_password = str(data.get("newPassword", ""))

    if not email or len(otp) != 6 or not otp.isdigit():
        return jsonify({"error": "Enter a valid email and six-digit OTP."}), 400

    if len(new_password) < 8:
        return jsonify({"error": "Password must contain at least 8 characters."}), 400

    valid, message = verify_otp(email, otp, "password_reset")

    if not valid:
        return jsonify({"error": message}), 400

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET password_hash = %s
                WHERE email = %s AND email_verified = TRUE
                """,
                (generate_password_hash(new_password), email),
            )

    return jsonify({"message": "Password reset successfully."})


@auth_bp.route("/me", methods=["GET"])
def current_user():
    authorization = request.headers.get("Authorization", "")

    if not authorization.startswith("Bearer "):
        return jsonify({"error": "Authentication required."}), 401

    token = authorization.split(" ", 1)[1].strip()
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT users.id, users.full_name, users.email
                FROM sessions
                JOIN users ON sessions.user_id = users.id
                WHERE sessions.token_hash = %s
                AND sessions.expires_at > NOW()
                AND users.email_verified = TRUE
                """,
                (token_hash,),
            )
            user = cursor.fetchone()

    if not user:
        return jsonify({"error": "Session expired."}), 401

    return jsonify({"user": user_response(user)})