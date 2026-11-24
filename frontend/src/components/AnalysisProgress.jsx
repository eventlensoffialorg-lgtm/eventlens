export default function AnalysisProgress({ status, progress = 0, stage = "" }) {
  if (!status || status === "idle") return null;
  const labels = { uploading: "Uploading video…", processing: "Analysing video…", completed: "Analysis completed", failed: "Analysis failed" };
  return <div className="progress-card"><div className="progress-heading"><strong>{status === "processing" && stage ? stage : labels[status] || status}</strong><span>{status === "uploading" || status === "processing" ? `${progress}%` : status === "completed" ? "Done" : ""}</span></div><div className="progress-track"><div className="progress-fill" style={{width:`${status === "completed" ? 100 : progress}%`}}/></div><small>{status === "uploading" ? "Sending your video to the analysis service." : status === "processing" ? "The backend is running the selected analysis stage." : status === "completed" ? "Your analysis result is ready." : "Check the backend connection and try again."}</small></div>;
}
