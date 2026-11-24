import { useCallback, useState } from "react";
import { getAnalysis, uploadVideoToBackend } from "../services/videoService";

export function useAnalysis() {
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [stage, setStage] = useState("");

  const analyze = useCallback(async (file) => {
    setStatus("uploading");
    setProgress(0);
    setResult(null);
    setError("");

    try {
      const uploadData = await uploadVideoToBackend(file, (event) => {
        if (event.total) {
          const percentage = Math.round(
            (event.loaded / event.total) * 100
          );

          setProgress(percentage);
        }
      });

      setStatus("processing");
      setProgress(uploadData.progress || 5);
      setStage(uploadData.stage || "Extracting frames");

      const jobId = uploadData.job_id || uploadData.report_id;
      if (!jobId) {
        throw new Error("The backend did not return an analysis job identifier.");
      }

      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        const job = await getAnalysis(jobId);

        if (job.status === "failed") {
          throw new Error(job.error || "Analysis failed.");
        }

        setStage(job.stage || "Processing video");
        if (typeof job.progress === "number") {
          setProgress(job.progress);
        }

        if (job.status === "completed") {
          setProgress(100);
          setStage("Completed");
          setResult(job);
          setStatus("completed");
          return job;
        }
      }
    } catch (err) {
      const message =
        err.response?.data?.message ||
        err.response?.data?.error ||
        err.message ||
        "Analysis request failed.";

      setError(message);
      setStatus("failed");

      throw err;
    }
  }, []);

  const reset = () => {
    setStatus("idle");
    setProgress(0);
    setResult(null);
    setError("");
    setStage("");
  };

  return {
    status,
    progress,
    result,
    error,
    stage,
    analyze,
    reset
  };
}