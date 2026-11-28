import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  ArrowRight,
  Eye,
  EyeOff,
  LockKeyhole,
  Mail,
  UserRound,
  ShieldCheck
} from "lucide-react";

import Brand from "../components/Brand";
import {
  completeRegistration,
  signIn,
  signUp,
  verifyEmailOtp
} from "../services/auth";
import { useAuth } from "../hooks/useAuth";

export default function Login() {
  const [tab, setTab] = useState(() =>
  new URLSearchParams(window.location.search).get("tab") === "register"
    ? "register"
    : "login"
);
  const [step, setStep] = useState("form");

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [verifiedEmail, setVerifiedEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [otp, setOtp] = useState("");

  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const { isConfigured, refreshSession } = useAuth();

  const navigate = useNavigate();
  const location = useLocation();

  function getError(err) {
    return err.response?.data?.error || err.message || "Something went wrong.";
  }

  async function handleLogin(e) {
    e.preventDefault();
    setError("");
    setSuccess("");
    setBusy(true);

    try {
      await signIn({ email, password });
      await refreshSession();

      navigate(
        location.state?.from?.pathname || "/dashboard",
        { replace: true }
      );
    } catch (err) {
      setError(getError(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleRegister(e) {
    e.preventDefault();
    setError("");
    setSuccess("");

    setBusy(true);

    try {
      await signUp({ fullName, email });

      setStep("otp");
      setSuccess("A six-digit verification code has been sent to your email.");
    } catch (err) {
      setError(getError(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleVerifyOtp(e) {
    e.preventDefault();
    setError("");
    setSuccess("");
    setBusy(true);

    try {
      const result = await verifyEmailOtp({ email, otp });
      setVerifiedEmail(result.email || email);
      setStep("password");
      setOtp("");
      setSuccess("Email verified. Create a password to finish registration.");
    } catch (err) {
      setError(getError(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleCompleteRegistration(e) {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (password.length < 8) {
      setError("Password must contain at least 8 characters.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setBusy(true);
    try {
      await completeRegistration({
        email: verifiedEmail,
        password
      });
      await refreshSession();

      navigate("/dashboard", { replace: true });
    } catch (err) {
      setError(getError(err));
    } finally {
      setBusy(false);
    }
  }

  function changeTab(selectedTab) {
    setTab(selectedTab);
    setStep("form");
    setError("");
    setSuccess("");
    setOtp("");
    setPassword("");
    setConfirmPassword("");
    setVerifiedEmail("");
  }

  return (
    <div className="auth-page">

      <section className="auth-visual">
        <Brand light />

        <div className="auth-visual-content">
          <span className="overline">
            <i /> YOUR EVENT, UNDERSTOOD
          </span>

          <h1>
            Make sense
            <br />
            of every <em>moment.</em>
          </h1>

          <p>
            Turn long-form event videos into clear, timestamped
            reports you can review and share.
          </p>

          <div className="auth-steps">
            <div>
              <b>01</b>
              <span>
                <strong>Detect key activity</strong>
                <small>Identify what happens on stage.</small>
              </span>
            </div>

            <div>
              <b>02</b>
              <span>
                <strong>Build the timeline</strong>
                <small>Keep moments and timestamps together.</small>
              </span>
            </div>

            <div>
              <b>03</b>
              <span>
                <strong>Generate a report</strong>
                <small>Make event footage easier to review.</small>
              </span>
            </div>
          </div>
        </div>

        <div className="visual-orbit" />
      </section>

      <section className="auth-form-side">
        <div className="auth-form-wrap">

          {step === "otp" ? (
            <>
              <div className="eyebrow">EMAIL VERIFICATION</div>

              <h2>Verify your email</h2>

              <p className="auth-subtitle">
                Enter the six-digit code sent to {email}.
              </p>

              <form className="form-stack" onSubmit={handleVerifyOtp}>

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

                {error && <p className="field-error">{error}</p>}

                {success && (
                  <p className="success-message">{success}</p>
                )}

                <button
                  className="primary-button full"
                  disabled={busy || otp.length !== 6}
                >
                  {busy ? "Verifying..." : "Verify and continue"}
                  <ArrowRight size={18} />
                </button>

              </form>

              <p className="auth-switch">
                <button
                  type="button"
                  className="text-button"
                  onClick={() => {
                    setStep("form");
                    setOtp("");
                    setError("");
                    setSuccess("");
                  }}
                >
                  Back to registration
                </button>
              </p>
            </>
          ) : step === "password" ? (
            <>
              <div className="eyebrow">EMAIL VERIFIED</div>

              <h2>Create a password</h2>

              <p className="auth-subtitle">
                Choose a password to finish creating your account.
              </p>

              <form
                onSubmit={handleCompleteRegistration}
                className="form-stack"
              >
                <label>
                  Password

                  <div className="input-wrap">
                    <LockKeyhole size={18} />

                    <input
                      type={showPassword ? "text" : "password"}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="At least 8 characters"
                      autoComplete="new-password"
                      minLength={8}
                      required
                    />

                    <button
                      type="button"
                      className="input-icon-button"
                      onClick={() => setShowPassword(!showPassword)}
                    >
                      {showPassword
                        ? <EyeOff size={18} />
                        : <Eye size={18} />}
                    </button>
                  </div>
                </label>

                <label>
                  Confirm password

                  <div className="input-wrap">
                    <LockKeyhole size={18} />

                    <input
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="Confirm your password"
                      autoComplete="new-password"
                      required
                    />
                  </div>
                </label>

                {error && <p className="field-error">{error}</p>}

                {success && (
                  <p className="success-message">{success}</p>
                )}

                <button
                  className="primary-button full"
                  disabled={busy || !isConfigured}
                >
                  {busy ? "Creating account..." : "Create account"}
                  <ArrowRight size={18} />
                </button>
              </form>
            </>
          ) : (
            <>
              <div className="eyebrow">WELCOME TO EVENTLENS</div>

              <h2>
                {tab === "login"
                  ? "Sign in to EventLens"
                  : "Create your account"}
              </h2>

              <p className="auth-subtitle">
                {tab === "login"
                  ? "Your event intelligence workspace is waiting."
                  : "Set up your private event analysis workspace."}
              </p>

              

              {tab === "login" ? (
                <form onSubmit={handleLogin} className="form-stack">

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

                  <label>
                    <span className="label-row">
                      Password
                      <Link to="/forgot-password">
                        Forgot password?
                      </Link>
                    </span>

                    <div className="input-wrap">
                      <LockKeyhole size={18} />

                      <input
                        type={showPassword ? "text" : "password"}
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Enter your password"
                        autoComplete="current-password"
                        required
                      />

                      <button
                        type="button"
                        className="input-icon-button"
                        onClick={() => setShowPassword(!showPassword)}
                      >
                        {showPassword
                          ? <EyeOff size={18} />
                          : <Eye size={18} />}
                      </button>
                    </div>
                  </label>

                  {error && (
                    <p className="field-error">{error}</p>
                  )}

                  <button
                    className="primary-button full"
                    disabled={busy || !isConfigured}
                  >
                    {busy ? "Signing in..." : "Sign in"}
                    <ArrowRight size={18} />
                  </button>

                </form>
              ) : (
                <form onSubmit={handleRegister} className="form-stack">

                  <label>
                    Full name

                    <div className="input-wrap">
                      <UserRound size={18} />

                      <input
                        type="text"
                        value={fullName}
                        onChange={(e) => setFullName(e.target.value)}
                        placeholder="Your name"
                        autoComplete="name"
                        required
                      />
                    </div>
                  </label>

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

                  {error && (
                    <p className="field-error">{error}</p>
                  )}

                  <button
                    className="primary-button full"
                    disabled={busy || !isConfigured}
                  >
                    {busy ? "Sending code..." : "Continue"}
                    <ArrowRight size={18} />
                  </button>

                </form>
              )}

              <p className="auth-switch">
                {tab === "login" ? (
                  <>
                    New to EventLens?{" "}
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => changeTab("register")}
                    >
                      Create an account
                    </button>
                  </>
                ) : (
                  <>
                    Already have an account?{" "}
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => changeTab("login")}
                    >
                      Sign in
                    </button>
                  </>
                )}
              </p>

              <div className="auth-foot">
                <LockKeyhole size={14} />
                Your details are protected by EventLens authentication.
              </div>
            </>
          )}

        </div>
      </section>

    </div>
  );
}