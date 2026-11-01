import os
import  certifi
from dotenv import load_dotenv
ssl_cert=os.environ.get('SSL_CERT_FILE')
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ['SSL_CERT_FILE'] = certifi.where()
elif not ssl_cert:
    os.environ['SSL_CERT_FILE'] = certifi.where()

load_dotenv()
from pydantic_settings import BaseSettings,SettingsConfigDict


class Settings(BaseSettings):
    pinecone_index:str="medical_agent_demo"
    groqai_chat_model:str="openai/gpt-oss-20b"
    groqai_vision_model:str="qwen/qwen3.8-27b"
  
    allowed_origins:str="http://localhost:8000,http://127.0.0.1:8000"
    model_config=SettingsConfigDict(env_file=".env",extra='ignore')


settings=Settings()



    

