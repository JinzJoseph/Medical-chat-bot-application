import base64
import io
from os import path
import secrets
from pathlib import Path
from fastapi import FastAPI, File,Form,Header, UploadFile, Request
from fastapi.responses import HTMLResponse,response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel,field
from groq import ChatGroq

from uuid import uuid4
from app.config import settings
from app.agent import graph

ROOT=Path(__file__).resolve().parent.parent
app=FastAPI(title="Medical Chatbot",version="0.1.0")
app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins.split(','),allow_credentials=False,allow_methods=['GET','POST'],allow_headers=["Authorization","Content-Type"])
app.mount("/static",StaticFiles(directory=ROOT/'static'),name="static")


