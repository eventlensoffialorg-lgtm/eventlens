import api from "./api";

export async function uploadVideoToBackend(file, onUploadProgress) {
  const body = new FormData();
  body.append("video", file);

  const { data } = await api.post("/analyze", body, {
    headers: {
      "Content-Type": "multipart/form-data"
    },
    onUploadProgress
  });

  return data;
}

export async function startAnalysis(payload) {
  const { data } = await api.post("/analyses", payload);
  return data;
}

export async function getAnalysis(id) {
  const { data } = await api.get(`/analyses/${id}`);
  
  return data;
}