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

  useEffect(() => {
    let active = true;

    getCurrentSession()
      .then((current) => {
        if (active) setSession(current);
      })
      .catch((error) => {
        console.error("Session lookup failed:", error);
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
    refreshSession
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