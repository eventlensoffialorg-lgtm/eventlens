import { useState } from "react";
import { AlertCircle, ArrowRight, Clapperboard, Play, RotateCcw, ScanLine, Download } from "lucide-react";
import AnalysisProgress from "../components/AnalysisProgress";
import DetectionViewer from "../components/DetectionViewer";
import EventTimeline from "../components/EventTimeline";
import GeneratedVideo from "../components/GeneratedVideo";
import PageHeading from "../components/PageHeading";
import VideoPreview from "../components/VideoPreview";
import VideoUploader from "../components/VideoUploader";
import { useAnalysis } from "../hooks/useAnalysis";
import { downloadReportPdf } from "../services/reportService";

export default function Analysis() {
  const [file, setFile] = useState(null);
  const [eventType, setEventType] = useState("Auto-detect");
  const analysis = useAnalysis();

  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState("");

  // Get analysis payload first
  const payload = analysis.result?.result || analysis.result || {};

  // Get report ID
  const reportId =
    analysis.result?.report_id ||
    analysis.result?.id ||
    payload?.id;

  const busy =
    analysis.status === "uploading" ||
    analysis.status === "processing";
async function handleDownloadReport() {
  if (!reportId) {
    setDownloadError("Report ID is unavailable.");
    return;
  }

  try {
    setDownloading(true);
    setDownloadError("");

    await downloadReportPdf(
      reportId,
      `EventLens-Report-${reportId}.pdf`
    );
  } catch (error) {
    setDownloadError(
      error.response?.data?.error || "Unable to download the PDF report."
    );
  } finally {
    setDownloading(false);
  }
}

  async function submit(event) {
    event.preventDefault();
    if (!file) return;
    try {
      await analysis.analyze(file);
    } catch {
      // The hook stores and displays the request error.
    }
  }

  

  return <>
    <PageHeading eyebrow="VIDEO WORKSPACE" title="Analyse an event" description="Upload a recording and prepare it for intelligent event analysis." />
    <div className="analysis-grid">
      <section className="panel">
        <div className="panel-heading">
          <div><span className="step-label">STEP 01</span><h2>Upload your video</h2><p>Choose a recorded cultural event, presentation or stage performance.</p></div>
          <span className="panel-icon"><Clapperboard size={20} /></span>
        </div>
        <form onSubmit={submit}>
          <VideoUploader onFileSelected={setFile} disabled={busy} />
          <label className="field-label event-type-label">Event category <span className="optional">Optional</span>
            <select className="select-control" value={eventType} onChange={(event) => setEventType(event.target.value)}>
              <option>Auto-detect</option><option>Cultural programme</option><option>Dance performance</option><option>Singing performance</option><option>Presentation</option><option>Speech</option><option>Panel discussion</option><option>Prize distribution</option>
            </select>
          </label>
          <button className="primary-button full" disabled={!file || busy}><Play size={16} /> Start video analysis <ArrowRight size={17} /></button>
        </form>
        <AnalysisProgress status={analysis.status} progress={analysis.progress} stage={analysis.stage} />
        {analysis.error && <div className="error-panel"><AlertCircle size={18} /><span>{analysis.error} Make sure the Flask backend is running and exposes the expected endpoint.</span></div>}
        {analysis.status !== "idle" && <button className="text-button" onClick={analysis.reset}><RotateCcw size={15} /> Reset analysis</button>}
      </section>
      <section className="panel">
        <div className="panel-heading"><div><span className="step-label">VIDEO PREVIEW</span><h2>Selected recording</h2><p>Check the video before sending it for analysis.</p></div></div>
        <VideoPreview file={file} />
        {file && <div className="video-meta"><span>{file.name}</span><small>{(file.size / (1024 * 1024)).toFixed(1)} MB</small></div>}
      </section>
    </div>

    <section className="section-block">
      <div className="section-title-row">
        <div><h2>Analysis output</h2><p>Detection frames, event summary and generated video.</p></div>
        <span className="status-pill neutral">{analysis.status === "completed" ? "Results received" : "Waiting for analysis"}</span>
      </div>
      <div className="report-download-area">
  <button
    className="primary-button"
    onClick={handleDownloadReport}
    disabled={downloading || !reportId}
  >
    <Download size={17} />
    {downloading ? "Downloading PDF..." : "Download Report (PDF)"}
  </button>

  {downloadError && (
    <p className="error-message">{downloadError}</p>
  )}
</div>
      {payload.summary && <div className="panel event-summary">
        <div>
          <span className="step-label">DETECTED EVENT</span>
          <h2>{payload.event_type || "Event summary"}</h2>
          <p>{payload.summary}</p>
          {payload.narrative && <p>{payload.narrative}</p>}
          {payload.peak_person_count != null && <p><strong>Peak visible people in a sampled frame:</strong> {payload.peak_person_count}. This is not a unique attendee count.</p>}
          {payload.unique_tracked_people != null && <p><strong>Unique tracked people:</strong> {payload.unique_tracked_people}</p>}
          {payload.tracking_status && payload.unique_tracked_people == null && <p><strong>Person tracking:</strong> Unavailable — no unique count could be verified.</p>}
          {payload.action_predictions?.length > 0 && <p><strong>Suggested visible actions:</strong> {payload.action_predictions.map((item) => item.confidence != null ? `${item.action_name} (${Math.round(item.confidence * 100)}%)` : `${item.action_name} (uncalibrated score ${item.model_score})`).join(", ")}</p>}
          {payload.model_quality && <small>Classifier: {payload.model_quality}</small>}
        </div>
        {payload.confidence != null && <span className="summary-confidence">{Math.round(payload.confidence * 100)}% confidence</span>}
      </div>}
      <div className="output-grid">
        <div className="panel">
          <div className="panel-heading">
  <div>
    <h3><ScanLine size={17} /> Object detections</h3>
    <p>YOLO object detection across the complete video.</p>
  </div>
</div>

<div className="panel-heading">
  <div>
    <h3>Annotated Detection Frame</h3>
    <p>YOLO object detection results from the analysed video.</p>
  </div>
</div>

<DetectionViewer
  imageUrl={payload.annotated_frame_url}
  detections={payload.detections || []}
/>

{payload.generated_video_url && (
  <>
    <div className="panel-heading">
      <div>
        <h3>Generated Detection Video</h3>
        <p>Video processed with object detection.</p>
      </div>
    </div>
    <GeneratedVideo videoUrl={payload.generated_video_url} />
  </>
)}
{payload.original_video_url && (
  <div className="panel">
    <div className="panel-heading">
      <h3>Original Uploaded Video</h3>
    </div>
    <GeneratedVideo videoUrl={payload.original_video_url} />
  </div>
)}
        </div>
      </div>
    </section>
  </>;
}
