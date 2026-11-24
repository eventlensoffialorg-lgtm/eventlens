import api from "./api";

export async function getReports() {
  const { data } = await api.get("/reports");
  return data;
}

export async function getReport(id) {
  const { data } = await api.get(`/reports/${id}`);
  return data;
}

export async function downloadReportPdf(
  id,
  filename = `EventLens-report-${id}.pdf`
) {
  const response = await api.get(`/reports/${id}/pdf`, {
    responseType: "blob"
  });

  const url = URL.createObjectURL(response.data);
  const link = document.createElement("a");

  link.href = url;
  link.download = filename;

  document.body.appendChild(link);
  link.click();
  link.remove();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
}