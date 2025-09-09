import React from 'react';
import Image from 'next/image';

// Classification result structure
interface ClassificationPrediction {
  class_name: string;
  confidence: number;
}

// Cropping result structure
interface CroppingResult {
  s3_url: string;
}

// Union type for all possible result types
type Result = ClassificationPrediction[] | CroppingResult | null

// Props for the ResultDisplay component
interface ResultDisplayProps {
  result: Result;
  loading: boolean;
  error: string | null;
}

// Type guard to check if result is a cropping result
function isCroppingResult(result: Result): result is CroppingResult {
  return result !== null && !Array.isArray(result) && 's3_url' in result;
}

// Displays processing results, loading state, or errors
export default function ResultDisplay({ result, loading, error }: ResultDisplayProps) {
  if (loading) {
    return <div className="text-center text-gray-900">Processing...</div>
  }

  if (error) {
    return <div className="text-red-700">{error}</div>
  }

  if (!result) {
    return null
  }

  // Classification result (array of predictions)
  if (Array.isArray(result)) {
    return (
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-black">Classification Results</h2>
        <div className="bg-gray-100 rounded-lg p-6 pt-12 space-y-12">
          {/* Debug output to see exactly what we're getting */}
          <pre className="bg-white p-4 rounded shadow text-sm overflow-auto text-black">
            {JSON.stringify(result, null, 2).split('\n').slice(1, -1).join('\n')}
          </pre>
        </div>
      </div>
    )
  }

  // Cropping result (contains s3_url)
  if (isCroppingResult(result)) {
    return (
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-black">Cropping Results</h2>
        <div className="bg-gray-100 rounded-lg p-6 pt-12 space-y-12">
          {/* Image display */}
          <div className="mb-4 flex justify-center items-center">
          <Image
            src={result.s3_url}
            alt="Processed result"
            width={800}
            height={600}
            className="max-w-[80%] h-auto rounded-lg"
            style={{ objectFit: 'contain' }}
          />
          </div>

          {/* Debug output to see exactly what we're getting */}
          <pre className="bg-white p-4 rounded shadow text-sm overflow-auto text-black">
            {JSON.stringify(result.s3_url, null, 2).slice(1, -1)}
          </pre>
        </div>
      </div>
    )
  }

  return null
}
