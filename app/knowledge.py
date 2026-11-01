from functools import lru_cache
from pinecone import Pinecone,ServerlessSpec
from openai import OpenAI
from app.config import settings
from groq import Groq
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')
pinecone_api_key=os.getenv('PINECONE_API_KEY')


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


@lru_cache()
def clients():
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set in environment variables.")

    groq_client = Groq(
        api_key=groq_api_key
    )

    pinecone_client = Pinecone(
        api_key=pinecone_api_key
    )

    return {
        "groq": groq_client,
        "pinecone": pinecone_client
    }

# def ensure_index():
#     client = clients()
#     pc = client["pinecone"]

#     if not pc.has_index(settings.pinecone_index):
#         pc.create_index(
#             name=settings.pinecone_index,
#             dimension=384,
#             metric="cosine",
#             spec=ServerlessSpec(
#                 cloud="aws",
#                 region="us-east-1"
#             )
#         )

#     return pc.Index(settings.pinecone_index)

def ensure_index():
    client = clients()
    pc = client["pinecone"]

    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pc.Index(settings.pinecone_index)

# index = ensure_index()

def embed(texts):
    vectors = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return vectors.tolist()

# def embed(texts):
#     ai,_=clients()
#     return [x.embedding for x in ai.embeddings.create(model="text-embedding-3-large",input=texts).data]


def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query])[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{'page': m.metadata.get('page'), 'text': m.metadata.get('text',''), 'score': round(m.score,3), 'source': 'Gale Encyclopedia (2002)'} for m in result.matches if m.score >= 0.30]