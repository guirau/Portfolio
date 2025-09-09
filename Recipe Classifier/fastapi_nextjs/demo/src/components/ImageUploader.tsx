import { useState, useRef, ChangeEvent, DragEvent, FormEvent } from 'react';
import Image from 'next/image';
import { Upload } from 'lucide-react';

// Type of image data that can be either a file or a URL
type ImageData = {
  type: 'file' | 'url';
  data: File | string;
} | null;

// Props for the ImageUploader component
interface ImageUploaderProps {
  setImage: (imageData: ImageData) => void;
  setError: (error: string | null) => void;
}

export default function ImageUploader({ setImage, setError }: ImageUploaderProps) {
  // State for URL input field
  const [url, setUrl] = useState<string>('');

  // UI states
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [isReady, setIsReady] = useState<boolean>(false);

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef<number>(0);

  /**
   * Processes a file, creates a preview, and updates parent state
   * @param file - The image file to process
   */
  const handleFile = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      const previewUrl = URL.createObjectURL(file);
      setPreview(previewUrl);
      setImage({
        type: 'file',
        data: file,
      });
      setError(null);
      setIsReady(false); // Reset isReady when uploading a file
    }
    else {
      setError('Please upload an image file');
      setPreview(null);
      setIsReady(false);
    }
  };

  // Handles file input change events
  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  // Handles URL form submission
  const handleUrlSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (url.trim()) {
      setImage({ type: 'url', data: url.trim() });
      setPreview(url.trim());
      setError(null);
      setIsReady(true);
    }
  };

  // Drag and drop handlers
  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    dragCounter.current = 0;

    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="space-y-6">
      <div
        role="button"
        tabIndex={0}
        className={`
          relative space-y-4 border-2 border-dashed rounded-lg p-6
          ${isDragging
            ? 'border-blue-500 bg-blue-100'
            : 'border-gray-300 hover:border-gray-400 bg-white'
          }
          transition-colors duration-200 cursor-pointer min-h-[200px]
          flex flex-col items-center justify-center
        `}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            fileInputRef.current?.click()
          }
        }}
      >
        {preview ? (
          <div className="relative w-full h-full flex items-center justify-center">
          <Image 
            src={preview}
            alt="Preview"
            width={400}
            height={300}
            className="max-h-[180px] max-w-full object-contain rounded"
            style={{ objectFit: 'contain' }}
          />
            <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-10 transition-opacity rounded flex items-center justify-center">
              <div className="text-transparent hover:text-white transition-colors">
                Click to change image
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Upload className="w-12 h-12 text-gray-400" />
            <div className="text-lg font-medium text-gray-700">
              Drop your image here
            </div>
            <div className="text-sm text-gray-500">
              or click to select a file
            </div>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
        />
      </div>

      <div className="flex items-center">
        <div className="flex-grow border-t border-gray-200"></div>
        <span className="mx-4 text-gray-500">or</span>
        <div className="flex-grow border-t border-gray-200"></div>
      </div>

      <div className="space-y-4">
        <form onSubmit={handleUrlSubmit} className="flex gap-2">
          <input
            type="url"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value)
              setIsReady(false) // Reset to 'Submit' when user types in text field
            }}
            placeholder="Enter image URL"
            className="flex-1 p-2 border rounded 
              bg-white
              text-gray-900
              border-gray-300
              focus:outline-none focus:ring-2 focus:ring-blue-500
              placeholder-gray-500"
          />
          <button
            type="submit"
            className={`px-4 py-2 text-white rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
              ${isReady
                ? 'bg-green-500 hover:bg-green-600'
                : 'bg-blue-500 hover:bg-blue-600'
              }`}
          >
            {isReady ? 'Ready' : 'Submit'}
          </button>
        </form>
      </div>
    </div>
  )
}
