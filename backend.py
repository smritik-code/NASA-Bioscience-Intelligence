# backend.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import os
from gtts import gTTS
import io
import base64
from typing import Optional

app = FastAPI(title="NASA Bioscience API", version="1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load precomputed data
DATA_DIR = "nasa_precomputed_data"

def load_data():
    """Load all precomputed data"""
    data = {}
    try:
        with open(f"{DATA_DIR}/publications.json", "r") as f:
            data['publications'] = json.load(f)
        with open(f"{DATA_DIR}/global_themes_network.json", "r") as f:
            data['global_network'] = json.load(f)
        with open(f"{DATA_DIR}/search_index.json", "r") as f:
            data['search_index'] = json.load(f)
        with open(f"{DATA_DIR}/metadata.json", "r") as f:
            data['metadata'] = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}

# Load data at startup
precomputed_data = load_data()

class SearchRequest(BaseModel):
    query: str
    search_type: str = "keyword"  # keyword, theme, technical_term

class AudioRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "NASA Bioscience API", "status": "running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "publications_loaded": len(precomputed_data.get('publications', {}))}

@app.get("/api/metadata")
async def get_metadata():
    return precomputed_data.get('metadata', {})

@app.get("/api/publications")
async def get_all_publications():
    return list(precomputed_data.get('publications', {}).keys())

@app.get("/api/publications/{pmc_id}")
async def get_publication(pmc_id: str):
    publications = precomputed_data.get('publications', {})
    if pmc_id in publications:
        return publications[pmc_id]
    raise HTTPException(status_code=404, detail="Publication not found")

@app.get("/api/global-network")
async def get_global_network():
    return precomputed_data.get('global_network', {})

@app.post("/api/search")
async def search_publications(request: SearchRequest):
    search_index = precomputed_data.get('search_index', {})
    results = set()

    if request.search_type == "theme":
        # Search by theme
        if request.query in search_index.get('by_theme', {}):
            results.update(search_index['by_theme'][request.query])

    elif request.search_type == "technical_term":
        # Search by technical term
        if request.query.lower() in search_index.get('by_technical_term', {}):
            results.update(search_index['by_technical_term'][request.query.lower()])

    else:  # keyword search
        # Search across all text fields
        query_words = request.query.lower().split()
        for word in query_words:
            if word in search_index.get('by_text', {}):
                results.update(search_index['by_text'][word])

    # Sort by relevance (publications with more technical terms first)
    publications = precomputed_data.get('publications', {})
    sorted_results = sorted(
        list(results),
        key=lambda x: len(publications.get(x, {}).get('technical_terms', [])),
        reverse=True
    )

    return {
        "query": request.query,
        "search_type": request.search_type,
        "results_count": len(sorted_results),
        "results": sorted_results[:50]  # Limit to 50 results
    }

@app.post("/api/audio/generate")
async def generate_audio(request: AudioRequest):
    """Generate audio from text"""
    try:
        if len(request.text) < 10:
            raise HTTPException(status_code=400, detail="Text too short")

        tts = gTTS(text=request.text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

        return {
            "audio_content": audio_base64,
            "text_length": len(request.text),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@app.get("/api/themes")
async def get_all_themes():
    """Get all available themes"""
    global_network = precomputed_data.get('global_network', {})
    themes = [node['id'] for node in global_network.get('nodes', [])]

    theme_descriptions = {
        'microgravity': 'Effects of weightlessness on biological systems',
        'radiation': 'Space radiation exposure and protection',
        'plant_biology': 'Plant growth and agriculture in space',
        'human_physiology': 'Human body adaptation to space environment',
        'microbiology': 'Microbial behavior and safety in space',
        'life_support': 'Environmental control and life support systems',
        'behavioral': 'Psychological and behavioral health',
        'technology': 'Space technology and instrumentation'
    }

    return {
        "themes": [
            {
                "id": theme,
                "name": theme.replace('_', ' ').title(),
                "description": theme_descriptions.get(theme, "NASA research theme"),
                "publications_count": len(global_network.get('theme_publications', {}).get(theme, []))
            }
            for theme in themes
        ]
    }

@app.get("/api/themes/{theme_id}/publications")
async def get_publications_by_theme(theme_id: str):
    """Get publications for a specific theme"""
    global_network = precomputed_data.get('global_network', {})
    theme_publications = global_network.get('theme_publications', {}).get(theme_id, [])

    return {
        "theme": theme_id,
        "publications_count": len(theme_publications),
        "publications": theme_publications[:100]  # Limit to 100
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
