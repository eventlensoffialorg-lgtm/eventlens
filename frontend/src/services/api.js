import axios from "axios";

const configuredApiUrl =
  import.meta.env.VITE_API_BASE_URL || "https://eventlens-xqy5.onrender.com/api";

const absoluteApiUrl = /^https?:\/\//i.test(configuredApiUrl)
  ? configuredApiUrl
  : `https://${configuredApiUrl}`;

const apiBaseUrl = absoluteApiUrl.replace(/\/+$/, "").endsWith("/api")
  ? absoluteApiUrl.replace(/\/+$/, "")
  : `${absoluteApiUrl.replace(/\/+$/, "")}/api`;

const api = axios.create({
  baseURL: apiBaseUrl,
  timeout: 300000
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("eventlens_token");

  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  return config;
});

export default api;