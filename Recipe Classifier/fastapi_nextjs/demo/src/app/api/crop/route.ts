import { NextRequest } from "next/server";

// Response structure from the Cropper API
interface CropResponse {
  s3_url?: string;
}

// Error response structure
interface ErrorResponse {
  error: string;
  detail?: string;
}

/**
 * Handles POST requests to crop images
 * @param request - The incoming HTTP request containing the image data
 * @returns JSON response with cropped image data or error message
 */
export async function POST(request: NextRequest): Promise<Response> {
  // Determine environment for API URL
  const ENV: string = process.env.ENVIRONMENT || "live";

  // Extract form data from the request
  const formData = await request.formData();
  const image = formData.get("file") as File | null;
  const url = formData.get("url") as string | null;

  // Create a new FormData object for the FastAPI request
  const apiFormData = new FormData();

  try {
    // If we have a direct file upload
    if (image) {
      // Pass through the uploaded file
      apiFormData.append("file", image);
    } else if (url) {
      // If we have a URL, fetch the image first
      const imageResponse = await fetch(url);
      if (!imageResponse.ok) throw new Error("Failed to fetch image from URL");

      // Extract filename from URL
      const originalFilename = new URL(url).pathname.split("/").pop() || "image.jpg";

      // Convert the reponse to a blob and append to form data
      const imageBlob = await imageResponse.blob();
      apiFormData.append("file", imageBlob, originalFilename);
    } else {
      // Return error if no image source is provided
      return Response.json(
        { error: "No image provided" } as ErrorResponse,
        { status: 400 }
      );
    }

    // Make request to the FastAPI classifier endpoint
    const response = await fetch(
      `https://recipecropper.${ENV}.company.com/api/v2/step/predict_upload`, {
      method: "POST",
      body: apiFormData,
    });

    // Log respone details for debugging
    console.log("Response status:", response.status);
    const responseText = await response.text();
    console.log("Cropping response:", responseText);

    // Handle API error responses
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}: ${responseText}`);
    }

    // Parse and return successful response
    const data = JSON.parse(responseText) as CropResponse
    return Response.json(data);

  } catch (error) {
    // Log and return any errors that occur during procesing
    console.error("Cropping error:", error);
    return Response.json(
      {
        error: "Cropping failed",
        details: error instanceof Error ? error.message : String(error)
      } as ErrorResponse, { status: 500 }
    );
  }
}
