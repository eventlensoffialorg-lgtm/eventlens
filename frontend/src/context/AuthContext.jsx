import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { getCurrentSession } from "../services/auth";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  async function refreshSession() {
    const current = await getCurrentSession();
    setSession(current);
    return current;
  }

  function setAuthenticatedSession(response) {
    if (!response?.token || !response?.user) {
      throw new Error("The server returned an invalid login session.");
    }

    const nextSession = { access_token: response.token, user: response.user };
    setSession(nextSession);
    return nextSession;
  }

  useEffect(() => {
    let active = true;

    getCurrentSession()
      .then((current) => {
        if (active) setSession(current);
      })
      .catch((error) => {
        console.error(
          "Could not verify the current EventLens session:",
          error?.response?.data?.error || error?.message || "Unexpected network error."
        );
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const value = useMemo(() => ({
    session,
    user: session?.user ?? null,
    loading,
    isConfigured: true,
    refreshSession,
    setAuthenticatedSession
  }), [session, loading]);

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider");
  }

  return context;
}