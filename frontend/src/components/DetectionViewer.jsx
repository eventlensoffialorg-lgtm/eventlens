import { useEffect, useState } from "react";
import { ScanLine } from "lucide-react";
import api from "../services/api";

export default function DetectionViewer({ imageUrl }) {
  const [protectedImageUrl, setProtectedImageUrl] = useState("");
  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    let active = true;
    let objectUrl = "";

    if (!imageUrl?.startsWith("/api/media/")) {
      setProtectedImageUrl(imageUrl || "");
      setImageError(false);
      return undefined;
    }

    setProtectedImageUrl("");
    setImageError(false);

    const apiPath = imageUrl.replace(/^\/api(?=\/)/, "");

    api.get(apiPath, { responseType: "blob" })
      .then(({ data }) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(data);
        setProtectedImageUrl(objectUrl);
      })
      .catch(() => {
        if (active) setImageError(true);
      });

    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [imageUrl]);

  const displayImageUrl = imageUrl?.startsWith("/api/media/")
    ? protectedImageUrl
    : imageUrl;

  return (
    <div className="detection-panel">
      {displayImageUrl ? (
        <div className="detection-image-wrap">
          <img src={displayImageUrl} alt="YOLO detected objects" />
        </div>
      ) : (
        <div className="empty-preview">
          <ScanLine size={28} />
          <span>
            {imageError
              ? "The annotated frame could not be loaded."
              : imageUrl
                ? "Loading annotated frame..."
                : "Annotated frames will appear here after analysis."}
          </span>
        </div>
      )}
    </div>
  );
}