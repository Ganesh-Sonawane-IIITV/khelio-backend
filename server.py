# server.py
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import build_graph
from back import random_id

load_dotenv()

app = FastAPI(title="Product Imagery Pipeline (LangGraph + Gemini)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph_app = build_graph()

class ProcessRequest(BaseModel):
    youtube_url: str
    save: bool = False
    save_dir: Optional[str] = None  # if save=True and not provided, we'll create out/<job_id>

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/process")
def process(req: ProcessRequest):
    state = {"youtube_url": req.youtube_url}
    if req.save:
        state["save_dir"] = req.save_dir or os.path.join("out", random_id())

    try:
        result = graph_app.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "youtube_url": req.youtube_url,
        "save_dir": result.get("save_dir"),
        "products": result.get("products", []),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)