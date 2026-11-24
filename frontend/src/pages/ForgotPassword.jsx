import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight, Mail, LockKeyhole, ShieldCheck } from "lucide-react";
import Brand from "../components/Brand";
import { sendPasswordReset, resetPassword } from "../services/auth";

export default function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [otpSent, setOtpSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const navigate = useNavigate();

  async function handleSendOtp(e) {
    e.preventDefault();
    setError("");
    setSuccess("");
    setBusy(true);

    try {
      const result = await sendPasswordReset(email);
      setOtpSent(true);
      setSuccess(result.message || "If the account exists, a verification code has been sent.");
    } catch (err) {
      setError(err.response?.data?.error || err.message || "Unable to send verification code.");
    } finally {
      setBusy(false);
    }
  }

  async function handleResetPassword(e) {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (newPassword.length < 8) {
      setError("Password must contain at least 8 characters.");
      return;
    }

    if (newPassword !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setBusy(true);

    try {
      const result = await resetPassword({
        email,
        otp,
        newPassword
      });

      setSuccess(result.message || "Password reset successfully.");

      setTimeout(() => {
        navigate("/login", { replace: true });
      }, 1500);
    } catch (err) {
      setError(err.response?.data?.error || err.message || "Unable to reset password.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-page">
      <section className="auth-visual">
        <Brand light />

        <div className="auth-visual-content">
          <span className="overline">
            <i /> EVENTLENS SECURITY
          </span>

          <h1>
            Your account,
            <br />
            <em>protected.</em>
          </h1>

          <p>
            Reset your password securely using the verification code
            sent to your registered email address.
          </p>
        </div>

        <div className="visual-orbit" />
      </section>

      <section className="auth-form-side">
        <div className="auth-form-wrap">
          <div className="eyebrow">ACCOUNT RECOVERY</div>

          <h2>Forgot password?</h2>

          <p className="auth-subtitle">
            {otpSent
              ? "Enter the verification code and choose a new password."
              : "Enter your registered email to receive a verification code."}
          </p>

          {!otpSent ? (
            <form onSubmit={handleSendOtp} className="form-stack">
              <label>
                Email address

                <div className="input-wrap">
                  <Mail size={18} />

                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    autoComplete="email"
                    required
                  />
                </div>
              </label>

              {error && <p className="field-error">{error}</p>}
              {success && <p className="success-message">{success}</p>}

              <button className="primary-button full" disabled={busy}>
                {busy ? "Sending..." : "Send verification code"}
                <ArrowRight size={18} />
              </button>
            </form>
          ) : (
            <form onSubmit={handleResetPassword} className="form-stack">
              <label>
                Verification code

                <div className="input-wrap">
                  <ShieldCheck size={18} />

                  <input
                    type="text"
                    inputMode="numeric"
                    maxLength={6}
                    value={otp}
                    onChange={(e) =>
                      setOtp(e.target.value.replace(/\D/g, "").slice(0, 6))
                    }
                    placeholder="Enter 6-digit OTP"
                    required
                  />
                </div>
              </label>

              <label>
                New password

                <div className="input-wrap">
                  <LockKeyhole size={18} />

                  <input
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="At least 8 characters"
                    minLength={8}
                    autoComplete="new-password"
                    required
                  />
                </div>
              </label>

              <label>
                Confirm new password

                <div className="input-wrap">
                  <LockKeyhole size={18} />

                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm new password"
                    autoComplete="new-password"
                    required
                  />
                </div>
              </label>

              {error && <p className="field-error">{error}</p>}
              {success && <p className="success-message">{success}</p>}

              <button className="primary-button full" disabled={busy || otp.length !== 6}>
                {busy ? "Resetting..." : "Reset password"}
                <ArrowRight size={18} />
              </button>
            </form>
          )}

          <p className="auth-switch">
            <Link to="/login">
              <ArrowLeft size={14} /> Back to Sign In
            </Link>
          </p>

          <div className="auth-foot">
            <LockKeyhole size={14} />
            Your account security matters to EventLens.
          </div>
        </div>
      </section>
    </div>
  );
}