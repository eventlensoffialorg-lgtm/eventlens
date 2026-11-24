import api from "./api";

const TOKEN_KEY = "eventlens_token";

export async function signUp({ fullName, email, password }) {
  const { data } = await api.post("/auth/register", {
    fullName,
    email,
    password
  });

  return data;
}

export async function verifyEmailOtp({ email, otp }) {
  const { data } = await api.post("/auth/verify-otp", {
    email,
    otp
  });

  if (data.token) {
    localStorage.setItem(TOKEN_KEY, data.token);
  }

  return data;
}

export async function signIn({ email, password }) {
  const { data } = await api.post("/auth/login", {
    email,
    password
  });

  if (data.token) {
    localStorage.setItem(TOKEN_KEY, data.token);
  }

  return data;
}

export async function signOut() {
  try {
    await api.post("/auth/logout");
  } finally {
    localStorage.removeItem(TOKEN_KEY);
  }
}

export async function sendPasswordReset(email) {
  const { data } = await api.post("/auth/forgot-password", {
    email
  });

  return data;
}

export async function resetPassword({ email, otp, newPassword }) {
  const { data } = await api.post("/auth/reset-password", {
    email,
    otp,
    newPassword
  });

  return data;
}

export async function getCurrentSession() {
  const token = localStorage.getItem(TOKEN_KEY);

  if (!token) {
    return null;
  }

  try {
    const { data } = await api.get("/auth/me");

    return {
      access_token: token,
      user: data.user
    };
  } catch {
    localStorage.removeItem(TOKEN_KEY);
    return null;
  }
}