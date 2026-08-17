import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import './New.css';


const SafetyMonitoringSystem = () => {
  const [model, setModel] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [debugInfo, setDebugInfo] = useState([]);
  const [capturedImage, setCapturedImage] = useState(null);
  const [detectionHistory, setDetectionHistory] = useState([]);
  const [isAlertVisible, setIsAlertVisible] = useState(false);
  const [alertType, setAlertType] = useState(null);
  const [previousDetections, setPreviousDetections] = useState({});


  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);



  // Telegram API details
  const TELEGRAM_BOT_TOKEN = '7956007164:AAGl4W2KoXOuWEAW60hXGnvGbcfaf-hhsBI';
  const TELEGRAM_CHAT_IDS = ['1363161353', '974623783'];

  // Adjusted thresholds for better accuracy
  const CROWD_THRESHOLD = 3;
  const CONFIDENCE_THRESHOLD = 0.6;
  const CROWD_DENSITY_THRESHOLD = 0.25;
  const FIRE_COLOR_THRESHOLD = 0.7; // Threshold for detecting fire-like colors
  const FIRE_PIXEL_THRESHOLD = 0.02;



  const ADDITIONAL_OBJECTS = ['mobile phone', 'book', 'handbag', 'backpack', 'shoe', 'laptop'];

  const logDebug = (message) => {
    setDebugInfo((prev) => [...prev.slice(-50), `${new Date().toLocaleTimeString()}: ${message}`]);
    console.log(message);
  };

  // Initialize TensorFlow and load model
  // Initialize TensorFlow and load model
  useEffect(() => {
    const loadModel = async () => {
      try {
        logDebug('Initializing TensorFlow.js...');
        await tf.ready();
        logDebug('Loading COCO-SSD model...');
        const loadedModel = await cocoSsd.load({
          base: 'mobilenet_v2'
        });
        setModel(loadedModel);
        setIsLoading(false);
        logDebug('Model loaded successfully');
      } catch (err) {
        const errorMessage = `Model loading failed: ${err.message}`;
        setError(errorMessage);
        logDebug(errorMessage);
        setIsLoading(false);
      }
    };

    loadModel();
  }, []);

  // Start webcam
  const startWebcam = async () => {
    try {
      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'environment'
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          if (canvasRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
          }
          if (captureCanvasRef.current) {
            captureCanvasRef.current.width = videoRef.current.videoWidth;
            captureCanvasRef.current.height = videoRef.current.videoHeight;
          }
          startDetection();
        };
      }
    } catch (err) {
      const errorMessage = `Webcam access failed: ${err.message}`;
      setError(errorMessage);
      logDebug(errorMessage);
    }
  };


  // Enhanced fire detection method
  const detectFire = () => {
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas) return false;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    let firePixelCount = 0;
    const totalPixels = canvas.width * canvas.height;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];

      if (r > 200 && g < 150 && b < 100) {
        firePixelCount++;
      }
    }

    const firePercentage = firePixelCount / totalPixels;
    logDebug(`Fire detection: ${firePercentage.toFixed(2)}%`);
    return firePercentage > FIRE_PIXEL_THRESHOLD;
  };


  // Modified detectIncident method to include enhanced fire detection
  const detectIncident = async (predictions) => {
    // People detection
    const people = predictions.filter(pred =>
      pred.class === 'person' && pred.score > CONFIDENCE_THRESHOLD
    );

    // Vehicle detection
    const vehicles = predictions.filter(pred =>
      ['car', 'truck', 'motorcycle', 'bus', 'bicycle'].includes(pred.class) &&
      pred.score > CONFIDENCE_THRESHOLD
    );

    // Fire indicators from object detection
    const fireIndicators = predictions.filter(pred =>
      ['fire hydrant', 'fire truck'].includes(pred.class) &&
      pred.score > CONFIDENCE_THRESHOLD
    );

    // Additional objects detection
    const additionalObjects = predictions.filter(pred =>
      ADDITIONAL_OBJECTS.includes(pred.class) &&
      pred.score > CONFIDENCE_THRESHOLD
    );

    // Crowd detection
    if (people.length >= CROWD_THRESHOLD) {
      const crowdDensity = calculateCrowdDensity(people);
      if (crowdDensity > CROWD_DENSITY_THRESHOLD) {
        return {
          type: 'overcrowd',
          confidence: Math.max(...people.map(p => p.score)),
          count: people.length,
          boxes: people.map(p => p.bbox)
        };
      }
    }

    // Accident detection
    if (vehicles.length >= 2) {
      const collisionPairs = detectCollision(vehicles);
      if (collisionPairs.length > 0) {
        return {
          type: 'accident',
          confidence: Math.max(...vehicles.map(v => v.score)),
          boxes: vehicles.map(v => v.bbox)
        };
      }
    }

    // Enhanced fire detection
    if (fireIndicators.length > 0 || detectFire()) {
      return { type: 'fire', confidence: 0.8 };
    }

    // Additional objects detection
    if (additionalObjects.length > 0) {
      return {
        type: 'object_movement',
        confidence: Math.max(...additionalObjects.map(o => o.score)),
        objects: additionalObjects.map(o => o.class),
        boxes: additionalObjects.map(o => o.bbox)
      };
    }

    return null;
  };

  // Calculate crowd density
  const calculateCrowdDensity = (people) => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;

    const totalArea = canvas.width * canvas.height;
    const peopleArea = people.reduce((acc, person) => {
      const [, , width, height] = person.bbox;
      return acc + (width * height);
    }, 0);

    return peopleArea / totalArea;
  };

  // Detect collision between vehicles
  const detectCollision = (vehicles) => {
    const collisionPairs = [];

    for (let i = 0; i < vehicles.length; i++) {
      for (let j = i + 1; j < vehicles.length; j++) {
        if (checkBoxOverlap(vehicles[i].bbox, vehicles[j].bbox)) {
          collisionPairs.push([vehicles[i], vehicles[j]]);
        }
      }
    }

    return collisionPairs;
  };

  // Check bounding box overlap
  const checkBoxOverlap = (bbox1, bbox2) => {
    const [x1, y1, w1, h1] = bbox1;
    const [x2, y2, w2, h2] = bbox2;

    return !(x1 + w1 < x2 ||
      x2 + w2 < x1 ||
      y1 + h1 < y2 ||
      y2 + h2 < y1);
  };

  // Improved object detection
  const detectObjects = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !model || video.readyState < 2) return;

    try {
      // Capture the current video frame
      const predictions = await model.detect(video);
      const incident = await detectIncident(predictions);

      setDetections(predictions);

      // Clear previous drawings
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw detected objects
      predictions.forEach((prediction) => {
        const [x, y, width, height] = prediction.bbox;

        // Draw bounding box
        ctx.beginPath();
        ctx.lineWidth = 3;
        ctx.strokeStyle = getDetectionColor(prediction.class, 0.7);
        ctx.rect(x, y, width, height);
        ctx.stroke();

        // Prepare label
        const label = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;
        ctx.font = '14px Arial';
        const labelWidth = ctx.measureText(label).width;

        // Draw label background
        ctx.fillStyle = getDetectionColor(prediction.class, 0.8);
        ctx.fillRect(x, y - 20, labelWidth + 10, 20);

        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(label, x + 5, y - 5);
      });

      // Handle any incidents
      if (incident) {
        handleIncident(incident);
      }
    } catch (error) {
      console.error('Object detection error:', error);
    }
  };




  // Get color based on object detection type
  const getDetectionColor = (className, alpha = 1) => {
    switch (className) {
      case 'person': return `rgba(0, 255, 0, ${alpha})`;
      case 'car':
      case 'truck':
      case 'motorcycle':
      case 'bus': return `rgba(255, 0, 0, ${alpha})`;
      case 'fire hydrant':
      case 'fire truck': return `rgba(255, 106, 0, ${alpha})`;
      case 'mobile phone':
      case 'book':
      case 'handbag':
      case 'backpack':
      case 'shoe':
      case 'laptop': return `rgba(128, 0, 128, ${alpha})`; // Purple for additional objects
      default: return `rgba(0, 0, 255, ${alpha})`;
    }
  };


  const sendToTelegram = async (incident) => {
    const message = `
🚨 *Incident Detected* 🚨
Type: ${incident.type}
Confidence: ${incident.confidence}%
Location: ${incident.location || 'Not available'}
Details: ${incident.details}
Time: ${incident.timestamp}
    `;

    // Send the message to all chat IDs
    for (const chatId of TELEGRAM_CHAT_IDS) {
      // Send text message
      await fetch(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: chatId,
          text: message,
          parse_mode: 'Markdown',
        }),
      });

      // Send the image
      if (incident.image) {
        const formData = new FormData();
        formData.append('chat_id', chatId);
        formData.append('photo', dataURItoBlob(incident.image), 'incident.jpg');

        await fetch(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendPhoto`, {
          method: 'POST',
          body: formData,
        });
      }
    }
  };

  // Helper function to convert data URI to Blob
  const dataURItoBlob = (dataURI) => {
    const byteString = atob(dataURI.split(',')[1]);
    const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const uint8Array = new Uint8Array(arrayBuffer);
    for (let i = 0; i < byteString.length; i++) {
      uint8Array[i] = byteString.charCodeAt(i);
    }
    return new Blob([uint8Array], { type: mimeString });
  };




  // Handle detected incidents
  const handleIncident = (incident) => {
    const timestamp = new Date().toLocaleTimeString();
    const image = captureVideoFrame();
    const defaultLocation = 'Bangalore, Karnataka';

    if (image) {
      const incidentDetails = {
        timestamp,
        image,
        type: incident.type,
        confidence: Math.round(incident.confidence * 100),
        location: defaultLocation,
        details:
          incident.type === 'overcrowd'
            ? `${incident.count} people detected`
            : incident.type === 'accident'
              ? 'Multiple vehicles detected'
              : incident.type === 'object_movement'
                ? `Detected: ${incident.objects.join(', ')}`
                : 'Fire-related objects detected',
      };

      setDetectionHistory((prev) => [incidentDetails, ...prev.slice(0, 9)]);
      setIsAlertVisible(true);
      setAlertType(incident.type);
      setCapturedImage(image);

      // Send incident details to Telegram
      sendToTelegram(incidentDetails);
    }
  };

  // Start continuous detection
  const startDetection = () => {
    if (!model || !videoRef.current) return;

    // Ensure canvas matches video dimensions
    const canvas = canvasRef.current;
    const video = videoRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const detectFrame = async () => {
      await detectObjects();
      requestAnimationFrame(detectFrame);
    };

    detectFrame();
  };


  // Capture video frame
  const captureVideoFrame = () => {
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;

    if (!video || !canvas) return null;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
  };

  // Handle alert visibility
  useEffect(() => {
    if (isAlertVisible) {
      const timer = setTimeout(() => setIsAlertVisible(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [isAlertVisible]);

  // Get alert title based on incident type
  const getAlertTitle = (type) => {
    switch (type) {
      case 'fire': return '🔥 Fire Hazard Detected!';
      case 'overcrowd': return '👥 Overcrowding Detected!';
      case 'accident': return '🚨 Potential Accident Detected!';
      case 'object_movement': return '📱 Object Movement Detected!';
      default: return '⚠️ Incident Detected!';
    }
  };

  return (
    <div className="safety-monitoring-container">
      <div className="monitoring-wrapper">
        {/* Header */}
        <header className="monitoring-header">
          <div className="header-content">
            <h1 className="system-title">
              <span className="ai-icon">🤖</span>
              AI Safety Monitoring
            </h1>
            {error && (
              <div className="error-notification">
                {error}
              </div>
            )}
          </div>
        </header>

        {/* Main Content */}
        <div className="monitoring-main-content">
          {/* Video Container */}
          <div className="video-monitoring-section">
            {!videoRef.current?.srcObject && (
              <button
                onClick={startWebcam}
                className={`start-monitoring-btn ${isLoading ? 'btn-disabled' : 'btn-active'}`}
                disabled={isLoading}
              >
                {isLoading ? "Loading AI Model..." : "Start Monitoring"}
              </button>
            )}

            <div className="video-canvas-container">
              <video
                ref={videoRef}
                className="monitoring-video"
                autoPlay
                playsInline
                muted
              />
              <canvas
                ref={canvasRef}
                className="detection-overlay-canvas"
              />
            </div>
          </div>

          {/* Incident Alert */}
          {isAlertVisible && capturedImage && (
            <div className="incident-alert-container">
              <div className="incident-alert-content">
                <h3 className="alert-title">
                  {getAlertTitle(alertType)}
                </h3>
                <img
                  src={capturedImage}
                  alt="Detected Incident"
                  className="incident-capture-image"
                />
                <p className="alert-description">
                  {alertType === 'overcrowd'
                    ? 'Warning: Too many people detected in the area!'
                    : alertType === 'accident'
                      ? 'Warning: Potential vehicle collision detected!'
                      : 'Warning: Fire-related objects detected!'}
                </p>
              </div>
            </div>
          )}

          {/* Detection History */}
          {detectionHistory.length > 0 && (
            <div className="detection-history-container">
              <h3 className="history-title">Incident History</h3>
              <div className="history-grid">
                {detectionHistory.map((detection, index) => (
                  <div
                    key={index}
                    className="history-item"
                  >
                    <img
                      src={detection.image}
                      alt={`Detection at ${detection.timestamp}`}
                      className="history-item-image"
                    />
                    <div className="history-item-details">
                      <p>
                        <strong>Type:</strong> {detection.type}
                        <br />
                        <strong>Time:</strong> {detection.timestamp}
                        <br />
                        <strong>Location:</strong> {detection.location}
                        <br />
                        <strong>Confidence:</strong> {detection.confidence}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Hidden Canvas for Capture */}
      <canvas
        ref={captureCanvasRef}
        width={640}
        height={480}
        className="hidden-capture-canvas"
      />
    </div>
  );
};

export default SafetyMonitoringSystem;