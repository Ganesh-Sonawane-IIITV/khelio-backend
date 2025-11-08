unzip the folder

cd backend

change the gemini API key, in env


I have used the free tier one , therefore I did not get that mich effective output, but If used the premium version , surely we will get the best output

pip install -r requirements.txt


python server.py



.env

GOOGLE_AI_API_KEY=api_key_here
GEMINI_ANALYSIS_MODEL=gemini-1.5-flash
GEMINI_SEGMENT_MODEL=gemini-1.5-flash
GEMINI_IMAGE_MODEL=gemini-2.0-flash-exp
IMAGEN_MODEL_ID=imagen-3.0-generate




#Final README
# 🎬 YouTube Product Detection & Enhancement Pipeline

A full-stack application that automatically detects products in YouTube videos, extracts them with precise segmentation, and generates enhanced marketing shots with AI-generated backgrounds.

## 🚀 Overview

This system processes YouTube videos through a multi-stage AI pipeline:

1. **Video Processing** - Downloads and extracts frames from YouTube videos
2. **Product Detection** - Uses Google Gemini to identify products and select the best frames
3. **Precise Segmentation** - Extracts products with polygon masks or GrabCut algorithms
4. **AI Enhancement** - Generates professional backgrounds and composites final marketing shots

## 🏗️ Architecture

### Backend (FastAPI + LangGraph)
```
Frontend (React) → FastAPI → LangGraph Pipeline → AI Services
↓
extract → identify → segment → enhance → persist
```

### Frontend (React)
- Modern React with Vite
- Real-time processing status
- Base64 image rendering
- Responsive UI for results display

## 🛠️ Technologies

**Backend:**
- Python 3.10+
- FastAPI (REST API)
- LangGraph (workflow orchestration)
- Google Generative AI (Gemini, Imagen 3)
- OpenCV, Pillow, NumPy (image processing)
- yt_dlp (YouTube download)

**Frontend:**
- React 18+
- Vite (build tool)
- Modern CSS

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+
- Google AI Studio API key

## ⚡ Quick Start

### 1. Backend Setup

```bash
# Clone and setup backend
cd backend
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_AI_API_KEY    (if  possible add the premium version API key would work better)
```

### 2. Start Backend Server

```bash
uvicorn server:app --reload --port 8000
```

### 3. Frontend Setup

```bash
cd khelio-frontend

# Configure API endpoint
echo VITE_API_BASE=http://localhost:8000 > .env

# Install and start
npm install
npm run dev
```

### 4. Access Application

Open http://localhost:5173 and paste a YouTube URL to process.

## 🔧 Configuration

### Environment Variables (Backend)

```env
GOOGLE_AI_API_KEY=your_google_ai_api_key
GEMINI_ANALYSIS_MODEL=gemini-1.5-flash
GEMINI_SEGMENT_MODEL=gemini-1.5-flash
GEMINI_IMAGE_MODEL=gemini-2.0-flash-exp
IMAGEN_MODEL_ID=imagen-3.0-generate-001
```

### Environment Variables (Frontend)

```env
VITE_API_BASE=http://localhost:8000
```

## 📊 API Documentation

### POST `/api/process`

**Request:**
```json
{
"youtube_url": "https://www.youtube.com/watch?v=...",
"save": true,
"save_dir": "out/my_job_id"
}
```

**Response:**
```json
{
"youtube_url": "...",
"save_dir": "out/abcd12",
"products": [
{
"name": "Product Name",
"best_frame_index": 12,
"reason": "Selected frame is sharp and centered",
"confidence": 0.87,
"bbox": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4},
"frame_b64": "<base64 jpeg>",
"segmentation": {
"polygon": [{"x": 0.12, "y": 0.18}, ...],
"tight_bbox": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4},
"mask_b64": "<base64 png>",
"cropped_b64": "<base64 png>"
},
"enhanced": ["<base64 png>", "<base64 png>", "<base64 png>"]
}
]
}
```

### GET `/health`

Returns service status:
```json
{"status": "ok"}
```

## 🔄 Processing Pipeline

### 1. Video Input & Frame Extraction
- YouTube URL validation and download (≤720p)
- Frame extraction at ~1 FPS (capped to ~90 frames)
- Base64 JPEG encoding for model compatibility

### 2. Product Identification
- Batched frame analysis with Google Gemini
- Structured JSON response with:
- Product name and confidence score
- Best frame selection with reasoning
- Normalized bounding box coordinates
- Fallback: Sharpest frame selection via Laplacian variance

### 3. Image Segmentation
- Primary: Gemini segmentation for precise polygon extraction
- Fallback 1: GrabCut algorithm for edge-aware cutout
- Fallback 2: Rectangular crop using bounding box
- Output: PNG mask + RGBA product cutout

### 4. Image Enhancement
- Background generation priority:
1. Imagen 3 (high-quality AI backgrounds)
2. Gemini inline images (if available)
3. Procedural gradients (guaranteed fallback)
- Professional compositing with soft shadows
- 2-3 enhanced shots per product

### 5. Persistence
- Optional disk storage in `out/<job_id>/`
- Complete result documentation in `result.json`
- Base64 images for immediate UI display

## 🎯 Key Features

- **Smart Frame Selection**: AI-powered best frame detection
- **Precise Segmentation**: Polygon-level product extraction
- **AI Background Generation**: Multiple enhancement options
- **Comprehensive Fallbacks**: Robust error handling at every stage
- **Real-time Processing**: Live status updates and immediate results
- **Production Ready**: CORS, error handling, and scalability

## ⏱️ Performance Notes

- First run: 20-90 seconds (depends on video length and API latency)
- Subsequent runs: Faster with cached processing
- Automatic quality degradation ensures completion

## 🚨 Limitations & Considerations

- Free tier Google AI API may have token limits
- Imagen 3 access requires specific Google Cloud permissions
- Video processing limited to 720p for performance
- Maximum ~90 frames analyzed per video

## 🔍 Troubleshooting

**Common Issues:**
- CORS errors: Ensure backend allows `http://localhost:5173`
- API key errors: Verify Google AI Studio API key validity
- Processing timeout: Longer videos may take additional time
- Memory issues: Frame count automatically capped

## 📁 Project Structure

```
├── backend/
│ ├── server.py # FastAPI application
│ ├── pipeline.py # LangGraph workflow
│ ├── processors/ # Processing modules
│ └── requirements.txt
├── khelio-frontend/
│ ├── src/
│ │ ├── components/ # React components
│ │ └── App.jsx
│ └── package.json
└── out/ # Processing outputs (generated)
```

## 🎉 Demo

Try with product review videos or unboxing content for best results. The system excels at identifying consumer products with clear visual presentation.

---

**Development Time**: ~5 hours total
**Pipeline Research & Design**: 1 hour
**Backend Implementation**: 1 hour
**Testing & Edge Cases**: 3 hours

## Challenges faced

- To find the best model , since I was trying to get the free version only , it took a lot of time for testing because the tokens limit were exceeding.

**Gemini Utilization**
-Multimodal understanding for product detection and best frame selection.
-Structured JSON via response_schema for robust parsing.
-Segmentation prompts for polygon + tight bbox extraction.
-Optional image generation (if inline image capability is enabled in your account) for backgrounds.
