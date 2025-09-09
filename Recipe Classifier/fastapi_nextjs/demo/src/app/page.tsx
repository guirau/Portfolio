'use client'
import { useState } from 'react'
import ImageUploader from '@/components/ImageUploader'
import ImageProcessor from '@/components/ImageProcessor'
import ResultDisplay from '@/components/ResultDisplay'

// Represents the image data
// Can be ither a file or a URL
type ImageData = {
  type: 'file' | 'url'  // Type of image data
  data: File | string   // The file object or URL string
} | null

// Structure of the classification result from the API
type ClassifyResult = Array<{
  class_name: string  // Predicted class name
  confidence: number  // Confidence score
}>

// Structure of the crop result from the API
type CropResult = {
  s3_url: string  // URL to the cropped image in S3
}

// Union type for all possible result types
type Result = ClassifyResult | CropResult | null

/**
 * Home page component
 * Manages the state and layout for the demo page
 */
export default function Home() {
  const [image, setImage] = useState<ImageData>(null)     // State of uploaded/selected image
  const [result, setResult] = useState<Result>(null)      // State of the API result
  const [loading, setLoading] = useState(false)           // Loading state during API request
  const [error, setError] = useState<string | null>(null) // Error state for API request

  return (
    <main className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-4xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold text-center text-gray-800">
          Step Classify & Crop Demo
        </h1>

        {/* Image upload section */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <ImageUploader setImage={setImage} setError={setError} />
        </div>

        {/* Image processing controls */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <ImageProcessor
            image={image}
            setResult={setResult}
            setLoading={setLoading}
            setError={setError}
          />
        </div>

        {/* Result display section */}
        {(result || loading || error) && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <ResultDisplay
              result={result}
              loading={loading}
              error={error}
            />
          </div>
        )}
      </div>
    </main>
  )
}
