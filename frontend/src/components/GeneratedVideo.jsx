import { useEffect, useState } from "react";
import { Film } from "lucide-react";
import api from "../services/api";

export default function GeneratedVideo({ videoUrl }) {
  const [source, setSource] = useState("");
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;
    let objectUrl = "";

    if (
  !videoUrl?.startsWith("/api/media/") &&
  !videoUrl?.startsWith("/api/original/")
) {
      setSource(videoUrl || "");
      setFailed(false);
      return undefined;
    }

    setSource("");
    setFailed(false);
    const apiPath = videoUrl.replace(/^\/api(?=\/)/, "");
    api.get(apiPath, { responseType: "blob" })
      .then(({ data }) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(data);
        setSource(objectUrl);
      })
      .catch(() => {
        if (active) setFailed(true);
      });

    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [videoUrl]);

  if (!videoUrl) return null;
  if (failed) return <div className="empty-preview"><Film size={24}/><span>Generated video could not be loaded.</span></div>;
  if (!source) return <div className="empty-preview"><Film size={24}/><span>Loading generated video…</span></div>;

  return <video className="video-preview generated-video" src={source} controls preload="metadata"/>;
}