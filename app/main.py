
import os
import base64
import secrets
import wave
from pathlib import Path
from uuid import uuid4
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    UploadFile,
    Request,
    HTTPException
)

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from groq import Groq
from app.config import settings
from app.agent import graph
import base64
import io
import secrets
from pathlib import Path
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from uuid import uuid4
from app.config import settings
from app.agent import graph

# Environment variables
API_TOKEN = os.getenv("API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_VISION_MODEL = os.getenv("GROQ_API_VISION_MODEL")

PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
TAVILY_API_KEY=os.getenv('TAVILY_API_KEY')
# Initialize Groq client
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")

groq_client = Groq(api_key=GROQ_API_KEY)


# Application setup
ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Medical Chatbot",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"]
)

app.mount(
    "/static",
    StaticFiles(directory=ROOT / "static"),
    name="static"
)


# Request schema
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

    history: list[dict[str, str]] = Field(
        default_factory=list,
        max_length=10
    )

    image_data: str | None = None

class SpeakRequest(BaseModel):
    text: str


# Authentication
def auth(authorization: str | None = None):

    if API_TOKEN and not secrets.compare_digest(
        authorization or "",
        f"Bearer {API_TOKEN}"
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API token"
        )


# Image inspection using Groq Vision
def inspect_image(data: str | None):

    if not data:
        return ""

    try:
        header, payload = data.split(",", 1)

        mime = header.split(";")[0].split(":")[1]

        raw = base64.b64decode(
            payload,
            validate=True
        )

        if mime not in (
            "image/jpeg",
            "image/png",
            "image/webp"
        ):
            raise ValueError(
                "Only JPEG, PNG and WebP images are supported."
            )

        if len(raw) > 5_000_000:
            raise ValueError(
                "Image must be smaller than 5MB."
            )

    except (ValueError, IndexError) as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}"
        )

    if not GROQ_API_VISION_MODEL:
        raise HTTPException(
            status_code=500,
            detail="Vision model is not configured."
        )

    try:

        result = groq_client.chat.completions.create(

            model=GROQ_API_VISION_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe only objectively visible "
                                "medical-relevant features. "
                                "Do not diagnose diseases, estimate "
                                "severity, or prescribe treatment. "
                                "If the image is unclear, say so."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data
                            }
                        }
                    ]
                }
            ],

            temperature=0,
            max_tokens=300
        )

        return (
            result.choices[0].message.content or ""
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"Image analysis failed: {str(exc)}"
        )


# setting the routes
@app.get('/',response_class=HTMLResponse)
def home():
    return (ROOT/'templates'/'index.html').read_text(encoding='utf-8')

@app.get('/api/health')
def health():
    return {"status": "ok",'configured':bool(GROQ_API_KEY and PINECONE_API_KEY),'tavily':bool(TAVILY_API_KEY)}

@app.post('/api/chat')
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(default=None, alias='X-Conversation-ID')
):
    auth(authorization)

    if not GROQ_API_KEY or not PINECONE_API_KEY:
        raise HTTPException(503,'Set GROQ_API_KEY and PINECONE_API_KEY in .env')

    history=[x for x in payload.history if x.get('role') in ('user','assistant') and isinstance(x.get('content'),str)][:10]

    # SQLite persistent memory: reuse the same conversation thread
    thread_id = conversation_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        observation=inspect_image(payload.image_data)

        state=graph.invoke(
            {
                'question':payload.message,
                'history':history,
                'image_description':observation,
                'kb':[],
                'web':[],
                'answer':'',
                'route':''
            },
            config=config
        )

        return {
            'answer':state['answer'],
            'sources': [{'type':'pdf','page':x['page'],'score':x['score']} for x in state['kb']]+[{'type':'web','url':x['url'],'title':x['title']} for x in state['web']],
            'tools': ['pinecone']+(['tavily'] if state['route']=='web' and TAVILY_API_KEY else [])+(['vision'] if observation else []),
            'conversation_id':thread_id
        }

    except HTTPException:
        raise
    except Exception as exc:
        print('Chat error:',repr(exc),flush=True)
        raise HTTPException(502,'AI service failed. Check server logs, credentials, Pinecone index, and connectivity.')

@app.post('/api/transcribe')

async def transcribe(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None)
):
    auth(authorization)

    # Read audio file
    data = await file.read(10_000_001)

    if len(data) > 10_000_000:
        raise HTTPException(
            status_code=413,
            detail="Audio exceeds 10 MB"
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty"
        )

    # Prepare audio file
    filename = file.filename or "voice.webm"

    audio = (filename, data)

    try:
        result = await groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=audio,
            response_format="json",
            temperature=0.0
        )

        return {
            "text": result.text
        }

    except Exception as exc:
        print("Groq transcription error:", str(exc))

        raise HTTPException(
            status_code=502,
            detail="Speech transcription unavailable"
        )




@app.post('/api/speak')


def split_text(text, limit=190):
    print(text)
    words = text.split()
    chunks = []
    current = ""

    for word in words:
        if len(word) > limit:
            raise ValueError("A single word exceeds the TTS limit")

        candidate = f"{current} {word}".strip()

        if len(candidate) <= limit:
            current = candidate
        else:
            chunks.append(current)
            current = word

    if current:
        chunks.append(current)

    return chunks


def generate_speech(text):
    audio_chunks = []

    for chunk in split_text(text):
        speech = groq_client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="troy",
            input=chunk,
            response_format="wav"
        )

        audio_chunks.append(speech.content)

    # Combine WAV audio chunks
    output = io.BytesIO()

    with wave.open(output, "wb") as combined:
        for index, audio_data in enumerate(audio_chunks):
            with wave.open(io.BytesIO(audio_data), "rb") as part:
                if index == 0:
                    combined.setparams(part.getparams())

                combined.writeframes(part.readframes(part.getnframes()))

    return output.getvalue()


@app.post("/api/speak")
async def speak(
    payload: SpeakRequest,
    authorization: str | None = Header(default=None)
):
    auth(authorization)

    text = payload.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Missing text")

    try:
        audio_data = await run_in_threadpool(
            generate_speech,
            text
        )

        return Response(
            content=audio_data,
            media_type="audio/wav"
        )

    except Exception as exc:
        print("Groq TTS error:", str(exc))
        raise HTTPException(
            status_code=502,
            detail="Speech synthesis unavailable"
        )