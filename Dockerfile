# Deployment image for the self-learning agent's Streamlit web UI.
# Works on Render / Railway / Fly.io free tiers.
#
# Persistent memory in deployment: set QDRANT_URL + QDRANT_API_KEY
# (free Qdrant Cloud cluster) as environment variables on your host —
# config.py automatically switches from local Chroma to Qdrant when
# QDRANT_URL is present, so memories survive redeploys.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by chromadb/sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render/Railway inject $PORT; default to 8501 for local docker run
ENV PORT=8501
EXPOSE 8501

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true