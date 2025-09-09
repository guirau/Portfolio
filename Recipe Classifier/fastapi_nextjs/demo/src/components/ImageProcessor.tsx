import React from 'react';

// Type definition for image data
// Can be either a file upload or URL string
type ImageData = {
  type: 'file' | 'url';
  data: File | string;
} | null;

// Type definition for classification and cropping results
type Result = {
  // Classification result
  class_name: string;
  confidence: number;
  // Cropping result
  s3_url: string;
} | Array<{
  class_name: string;
  confidence: number;
}> | null;

// Props interface for the ImageProcessor component
interface ImageProcessorProps {
  image: ImageData;
  setResult: (result: Result) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

// Component that provides image processing functionality
// Handles sending images to classification and cropping endpoints
export default function ImageProcessor({
  image,
  setResult,
  setLoading,
  setError,
}: ImageProcessorProps) {
  /**
   * Process an image through the specified API endpoint
   * @param endpoint - The API endpoint to use ("classify" or "crop")
   */
  const processImage = async (endpoint: "classify" | "crop"): Promise<void> => {
    if (!image) return;

    setLoading(true);
    setError(null);

    try {
      let response: Response;

      if (endpoint === "crop") {
        if (image.type === "file") {
          const formData = new FormData();
          formData.append("file", image.data as File); // Cast to File type
          response = await fetch("/api/crop", {
            method: "POST",
            body: formData,
          });
        } else {
          const formData = new FormData();
          formData.append("url", image.data as string); // Cast to string type
          response = await fetch("/api/crop", {
            method: "POST",
            body: formData,
          });
        }
      } else if (endpoint === "classify") {
        const formData = new FormData();
        if (image.type === "file") {
          formData.append("image", image.data as File); // Field name is "image"
        } else {
          formData.append("url", image.data as string); // Cast to string type
        }
        response = await fetch("/api/classify", {
          method: "POST",
          body: formData,
        });
      } else {
        throw new Error(`Invalid endpoint: ${endpoint}`);
      }

      if (!response || !response.ok) {
        throw new Error(`Processing failed: ${response?.statusText || 'Unknown error'}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setError(error instanceof Error ? error.message : "An unknown error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <button
          onClick={() => processImage('classify')}
          disabled={!image}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
        >
          Classify Image
        </button>
        <button
          onClick={() => processImage('crop')}
          disabled={!image}
          className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50"
        >
          Crop Image
        </button>
      </div>
    </div>
  );
}


