import { NextRequest } from "next/server";

// Response structure from the Classifier API
interface ClassificationResponse {
  class: string;
  confidence: number;
}

// Error response structure
interface ErrorResponse {
  error: string;
  detail?: string;
}

/**
 * Handles POST requests to classify images
 * @param request - The incoming HTTP request containing the image data
 * @returns JSON response with classification results or error message
 */
export async function POST(request: NextRequest): Promise<Response> {
  // Determine environment for API URL
  const ENV: string = process.env.ENVIRONMENT || "live";

  // Extract form data from the request
  const formData = await request.formData();
  const image = formData.get("image") as File | null;
  const url = formData.get("url") as string | null;

  // Create a new FormData object for the FastAPI request
  const apiFormData = new FormData();

  try {
    // If we have a direct file upload
    if (image) {
      // FastAPI expects the field name to be 'file'
      apiFormData.append("file", image);
    } else if (url) {
      // If we have a URL, fetch the image first
      const imageResponse = await fetch(url);
      if (!imageResponse.ok) throw new Error("Failed to fetch image from URL");

      // Convert the reponse to a blob and append to form data
      const imageBlob = await imageResponse.blob();
      apiFormData.append("file", imageBlob, "image.jpg");
    } else {
      // Return error if no image source is provided
      return Response.json(
        { error: "No image provided" } as ErrorResponse,
        { status: 400 }
      );
    }

    // Make request to the FastAPI classifier endpoint
    const response = await fetch(
      `https://recipeclassifier.${ENV}.company.com/api/v1/predict`, {
      method: "POST",
      body: apiFormData,
    });

    // Log reponse details for debugging
    console.log("Classification response:", response.status);
    const responseText = await response.text();
    console.log("Classification response:", responseText);

    // Parse response as JSON
    let responseJson;
    try {
      responseJson = JSON.parse(responseText);
    } catch (parseError) {
      console.error("Failed to parse response as JSON:", parseError);
      throw new Error(`Invalid response format with status ${response.status}: ${responseText}`);
    }

    // Handle API error responses
    if (!response.ok) {
      const error = responseJson as ErrorResponse;
      throw new Error(error.detail || "Classification failed");
    }

    // Return successful predictions
    return Response.json(responseJson as ClassificationResponse);

  } catch (error) {
    // Log and return any errors that occur during procesing
    console.error("Classification error:", error);
    return Response.json(
      { error: error instanceof Error ? error.message : "Classification failed" } as ErrorResponse,
      { status: 500 }
    );
  }
}
