import { useEffect, useRef, useState, useCallback } from "react";
import * as faceapi from "@vladmandic/face-api";
import { Brain, Camera, CameraOff, Activity, AlertTriangle, Lightbulb, MoveDiagonal } from "lucide-react";

// ============================================================================
// CONSTANTS & CONFIGURATION
// ============================================================================
const BUFFER_SIZE = 30; // Rolling window of frames (approx 12 seconds / 30 frames at 400ms)
const MIN_FACE_AREA_PERCENT = 8; // Face must take up at least 8% of the camera
const LIGHTING_CHECK_INTERVAL = 30; // Check lighting every 30 frames
// DROPPED from 1280x720 to prevent VRAM overflow
const TARGET_RESOLUTION = { width: 640, height: 480 }; 

export default function WebcamScanner({ onEmotionDetected }) {
  // --- Refs ---
  const videoRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const loopRef = useRef(null);
  const isActiveRef = useRef(false);
  
  // Mathematical Matrix for Emotion Smoothing
  const confidenceMatrix = useRef([]); 
  const frameCounter = useRef(0);

  // --- UI State ---
  const [isInitializing, setIsInitializing] = useState(true);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  
  // --- Telemetry State ---
  const [currentEmotion, setCurrentEmotion] = useState("neutral");
  const [faceDetected, setFaceDetected] = useState(false);
  const [rawScores, setRawScores] = useState(null);
  
  // --- Diagnostic Warnings ---
  const [warnings, setWarnings] = useState({
    lowLight: false,
    tooFar: false,
    noFace: false,
  });

  const stopVideo = useCallback(() => {
    isActiveRef.current = false;
    setIsCameraActive(false);
    setFaceDetected(false);
    setWarnings({ lowLight: false, tooFar: false, noFace: false });
    
    if (loopRef.current) clearTimeout(loopRef.current);
    
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  // ============================================================================
  // 1. MODEL INITIALIZATION
  // ============================================================================
  useEffect(() => {
    const loadHeavyModels = async () => {
      try {
        // SECURE OFFLINE-FIRST FIX: Loading models locally from the public asset directory
        const MODEL_URL = "/models";
        
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        ]);
        setIsInitializing(false);
      } catch (error) {
        console.error("CRITICAL ERROR: Failed to load neural weights.", error);
        setCameraError("Failed to load AI models. Check network.");
      }
    };
    loadHeavyModels();

    // Cleanup physics loops on unmount
    return () => stopVideo();
  }, [stopVideo]);

  // ============================================================================
  // 2. HARDWARE PIPELINE
  // ============================================================================
  const startVideo = async () => {
    setCameraError("");
    try {
      // Force hardware to provide stream mapped to the lower resolution
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: TARGET_RESOLUTION.width },
          height: { ideal: TARGET_RESOLUTION.height },
          facingMode: "user" 
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsCameraActive(true);
        isActiveRef.current = true;
      }
    } catch (err) {
      console.error("Hardware pipeline failed:", err);
      setCameraError("Camera access denied or hardware missing.");
    }
  };


  // ============================================================================
  // 3. ENVIRONMENTAL DIAGNOSTICS (Lighting Calculation)
  // ============================================================================
  const analyzeLighting = () => {
    if (!videoRef.current || !hiddenCanvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = hiddenCanvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Scale down massively to save CPU cycles
    canvas.width = 64;
    canvas.height = 64;
    ctx.drawImage(video, 0, 0, 64, 64);
    
    const imageData = ctx.getImageData(0, 0, 64, 64).data;
    let brightnessSum = 0;
    
    for (let i = 0; i < imageData.length; i += 16) {
      // Perceived luminance formula
      brightnessSum += (0.299 * imageData[i] + 0.587 * imageData[i+1] + 0.114 * imageData[i+2]);
    }
    
    // Adjust division to match the new sampling count (1024 samples instead of 4096)
    const avgBrightness = brightnessSum / 1024;
    // If average brightness is below 40 (out of 255), the room is practically pitch black for an AI
    setWarnings(prev => ({ ...prev, lowLight: avgBrightness < 40 }));
  };

  // ============================================================================
  // 4. THE NEURAL CORE (Main Execution Loop)
  // ============================================================================
  const runInferenceLoop = async () => {
    // Escape hatch: Stop if unmounted or camera stopped
    if (!isActiveRef.current || !videoRef.current || videoRef.current.readyState !== 4) {
      if (isActiveRef.current) loopRef.current = setTimeout(runInferenceLoop, 400);
      return;
    }

    try {
      frameCounter.current += 1;

      // Run lighting diagnostics periodically
      if (frameCounter.current % LIGHTING_CHECK_INTERVAL === 0) {
        analyzeLighting();
      }

      // Execute Neural Detection (SSD MobileNet V1)
      const options = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.25 });
      const detection = await faceapi
        .detectSingleFace(videoRef.current, options)
        .withFaceExpressions();

      if (!isActiveRef.current) return;

      if (detection) {
        setFaceDetected(true);
        setWarnings(prev => ({ ...prev, noFace: false }));
        setRawScores(detection.expressions);

        // --- Geometric Analytics ---
        const box = detection.detection.box;
        const videoArea = videoRef.current.videoWidth * videoRef.current.videoHeight;
        const faceArea = box.width * box.height;
        const facePercent = (faceArea / videoArea) * 100;
        
        // If face is too small, AI confidence plummets. Warn the user.
        setWarnings(prev => ({ ...prev, tooFar: facePercent < MIN_FACE_AREA_PERCENT }));

        // --- Weighted Confidence Matrix ---
        // Push raw probabilities into the buffer, discarding "neutral" noise threshold
        const exp = detection.expressions;
        const filteredExp = {
          happy: exp.happy,
          sad: exp.sad,
          angry: exp.angry,
          surprised: exp.surprised,
          fear: exp.fear,
          neutral: exp.neutral > 0.85 ? exp.neutral : 0 // Punish weak neutral signals
        };

        confidenceMatrix.current.push(filteredExp);
        if (confidenceMatrix.current.length > BUFFER_SIZE) {
          confidenceMatrix.current.shift(); // Maintain rolling window
        }

        // Calculate the aggregate mathematical sum of all probabilities over the buffer period
        const aggregate = { happy: 0, sad: 0, angry: 0, surprised: 0, fear: 0, neutral: 0 };
        for (const frame of confidenceMatrix.current) {
          for (const key in aggregate) {
            aggregate[key] += frame[key];
          }
        }

        // Isolate the dominant emotional vector
        let dominantEmotion = "neutral";
        let maxAggregate = 0;
        for (const key in aggregate) {
          if (aggregate[key] > maxAggregate) {
            maxAggregate = aggregate[key];
            dominantEmotion = key;
          }
        }

        // If the dominant emotion is not neutral, and has achieved sufficient aggregate weight, lock it.
        // We require non-neutral emotions to hit an aggregate score of 1.5 across the buffer 
        // to prevent single-frame false positives.
        let finalOutput = "neutral";
        if (dominantEmotion !== "neutral" && maxAggregate > 1.5) {
          finalOutput = dominantEmotion;
        }

        setCurrentEmotion(prev => {
          if (prev !== finalOutput) {
            if (onEmotionDetected) onEmotionDetected(finalOutput);
            return finalOutput;
          }
          return prev;
        });

      } else {
        // Total Detection Failure
        setFaceDetected(false);
        setWarnings(prev => ({ ...prev, noFace: true }));
        setRawScores(null);
      }
    } catch {
      // Suppress frame drop noise
    }

    if (isActiveRef.current) {
      // SLOWED DOWN: Give the GPU's garbage collector time to breathe
      loopRef.current = setTimeout(runInferenceLoop, 400); 
    }
  };

  // Safe formatter for UI telemetry
  const formatScore = (val) => (val ? (val * 100).toFixed(1) : "0.0");

  // ============================================================================
  // 5. RENDER / HUD
  // ============================================================================
  return (
    <div className="bg-ink border border-azure-800 rounded-xl p-4 flex flex-col items-center shadow-soft relative overflow-hidden">
      {/* Invisible canvas for lighting calculations */}
      <canvas ref={hiddenCanvasRef} className="hidden" />

      {/* Header */}
      <div className="flex items-center justify-between w-full mb-3">
        <h3 className="text-azure-100 text-sm font-semibold flex items-center gap-2">
          <Brain size={16} className="text-azure-400" />
          Omni-Sensor
        </h3>
        <button
          onClick={isCameraActive ? stopVideo : startVideo}
          disabled={isInitializing}
          className={`p-2 rounded-full transition-colors ${
            isCameraActive ? "bg-red-500/20 text-red-400 hover:bg-red-500/30" : "bg-azure-500/20 text-azure-400 hover:bg-azure-500/30"
          } disabled:opacity-50`}
        >
          {isCameraActive ? <CameraOff size={16} /> : <Camera size={16} />}
        </button>
      </div>

      {/* Hardware Error Overlay */}
      {cameraError && (
        <div className="w-full bg-red-900/50 text-red-200 text-xs p-2 rounded mb-3 text-center border border-red-500/50">
          {cameraError}
        </div>
      )}

      {/* Video Container */}
      <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden flex items-center justify-center border border-azure-900/50">
        {isInitializing ? (
          <div className="text-azure-400/50 text-xs text-center flex flex-col items-center gap-2">
            <div className="w-5 h-5 border-2 border-azure-500/30 border-t-azure-400 rounded-full animate-spin"/>
            Loading SSD Neural Weights...
          </div>
        ) : !isCameraActive ? (
          <div className="text-azure-400/50 text-xs">Sensor Offline</div>
        ) : null}

        {/* Environmental Warnings Overlay */}
        {isCameraActive && (
          <div className="absolute top-2 left-2 right-2 flex flex-col gap-1 z-20">
            {warnings.noFace && (
              <div className="bg-red-500/80 text-white text-[10px] font-bold px-2 py-1 rounded flex items-center gap-1 backdrop-blur-sm animate-pulse shadow-lg">
                <AlertTriangle size={12}/> NO FACE IN FRAME
              </div>
            )}
            {warnings.lowLight && !warnings.noFace && (
              <div className="bg-orange-500/80 text-white text-[10px] font-bold px-2 py-1 rounded flex items-center gap-1 backdrop-blur-sm shadow-lg">
                <Lightbulb size={12}/> LOW LIGHTING DETECTED
              </div>
            )}
            {warnings.tooFar && !warnings.noFace && (
              <div className="bg-blue-500/80 text-white text-[10px] font-bold px-2 py-1 rounded flex items-center gap-1 backdrop-blur-sm shadow-lg">
                <MoveDiagonal size={12}/> MOVE CLOSER
              </div>
            )}
          </div>
        )}

        <video
          ref={videoRef}
          onPlay={runInferenceLoop}
          width="640"  // Matched to standard VGA
          height="480" // Matched to standard VGA
          className={`w-full h-full object-cover transform -scale-x-100 ${!isCameraActive ? 'opacity-0' : 'opacity-100'} transition-opacity duration-500`}
          muted
          playsInline
        />
      </div>

      {/* Telemetry Dashboard */}
      {isCameraActive && (
        <div className="w-full mt-3 space-y-2">
          {/* Output State */}
          <div className="text-xs text-azure-200 bg-azure-900/50 px-3 py-2 rounded-lg flex items-center justify-between border border-azure-800/50 shadow-inner">
            <span className="flex items-center gap-1.5 opacity-80">
              <Activity size={12} className={faceDetected ? "text-green-400 animate-pulse" : "text-red-400"}/>
              Engine Status:
            </span>
            <span className={`font-bold uppercase tracking-widest ${faceDetected ? 'text-azure-300' : 'text-red-400'}`}>
              {faceDetected ? currentEmotion : 'BLIND'}
            </span>
          </div>

          {/* Raw Diagnostic Matrix */}
          <div className="bg-black/40 rounded-lg p-2.5 text-[10px] text-azure-300/60 font-mono border border-black/50">
            <div className="flex items-center justify-between mb-1.5 border-b border-azure-800/30 pb-1.5">
              <span>Raw Neural Output</span>
              <span className="text-azure-500">{faceDetected ? 'LIVE' : 'WAITING'}</span>
            </div>
            {rawScores ? (
              <div className="grid grid-cols-2 gap-y-1 gap-x-2">
                <div className="text-azure-200/50">N: {formatScore(rawScores.neutral)}%</div>
                <div className={rawScores.happy > 0.1 ? "text-green-400 font-bold drop-shadow-[0_0_2px_rgba(74,222,128,0.5)]" : ""}>H: {formatScore(rawScores.happy)}%</div>
                <div className={rawScores.sad > 0.1 ? "text-blue-400 font-bold drop-shadow-[0_0_2px_rgba(96,165,250,0.5)]" : ""}>S: {formatScore(rawScores.sad)}%</div>
                <div className={rawScores.angry > 0.1 ? "text-red-400 font-bold drop-shadow-[0_0_2px_rgba(248,113,113,0.5)]" : ""}>A: {formatScore(rawScores.angry)}%</div>
                <div className={rawScores.surprised > 0.1 ? "text-yellow-400 font-bold drop-shadow-[0_0_2px_rgba(250,204,21,0.5)]" : ""}>U: {formatScore(rawScores.surprised)}%</div>
                <div className={rawScores.fear > 0.1 ? "text-purple-400 font-bold drop-shadow-[0_0_2px_rgba(192,132,252,0.5)]" : ""}>F: {formatScore(rawScores.fear)}%</div>
              </div>
            ) : (
              <div className="py-2 text-center text-azure-500/40 italic">Awaiting tensor data...</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}