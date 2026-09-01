# Self-Learning AI Agent

A personal chat assistant using Groq for replies and mem0 for persistent,
semantic memory. It has both a terminal interface and a password-protected
Streamlit web interface.

## Important privacy note

This is a **single-owner/private** application. The Streamlit password gate
prevents casual public access but is not a replacement for per-user accounts.
Do not use this starter project to store health, financial, or other highly
sensitive data. Do not share its URL or `APP_PASSWORD` with untrusted people.

## Requirements

- Python 3.10+ (Python 3.11 is used by the Docker image)
- A free Groq API key

## Local setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

If you previously installed dependencies, recreate the virtual environment so
the pinned Torch and Transformers versions replace incompatible newer ones:

```powershell
deactivate
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Edit `.env` and set:

```env
GROQ_API_KEY=your_groq_key
APP_PASSWORD=a-long-unique-password
```

Leave the Qdrant values blank for local storage. The local Chroma database is
created in `memory_db/` and is deliberately ignored by Git.

## Run locally

Terminal interface:

```powershell
python chat.py
```

Web interface:

```powershell
streamlit run app.py
```

Open the URL Streamlit prints, normally `http://localhost:8501`, and enter
`APP_PASSWORD` to unlock the app. Use a user ID containing only letters,
numbers, hyphens, and underscores. The CLI commands are `/memories`,
`/forget`, and `/exit`.

## Deploy free with Streamlit Community Cloud + Qdrant Cloud

This is the recommended deployment route. Streamlit hosts the UI and Qdrant
keeps memory outside the host's temporary filesystem.

1. Create a free Qdrant Cloud cluster at https://cloud.qdrant.io. Copy its
   HTTPS URL and create an API key.
2. Create a new GitHub repository. Do **not** add `.env`, `venv`, or
   `memory_db`; `.gitignore` excludes them.
3. Push this project to GitHub:

   ```powershell
   git init
   git add .
   git commit -m "Prepare self-learning agent for deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
   git push -u origin main
   ```

4. Open https://share.streamlit.io, select the repository, choose branch
   `main`, and set `app.py` as the entrypoint.
5. Before deploying, open **Advanced settings** and add these secrets:

   ```toml
   GROQ_API_KEY = "your_groq_key"
   APP_PASSWORD = "a-long-unique-password"
   QDRANT_URL = "https://your-cluster.cloud.qdrant.io"
   QDRANT_API_KEY = "your_qdrant_key"
   ```

6. Click **Deploy**. Keep the generated URL private.

All four deployment secrets are required. The app rejects a partial Qdrant
configuration rather than silently using temporary storage.

## Docker / Render alternative

The included `Dockerfile` can deploy on Render. Choose a free Web Service,
connect the GitHub repository, and add the same four environment variables in
Render's Environment settings. Its filesystem is temporary, so Qdrant is
required. Free Render services can sleep when idle; Streamlit Community Cloud
is usually simpler for this project.

To test Docker locally:

```powershell
docker build -t self-learning-agent .
docker run --rm -p 8501:8501 --env-file .env self-learning-agent
```

## Project files

- `app.py` — password-protected Streamlit frontend
- `chat.py` — terminal frontend
- `memory_agent.py` — Groq + mem0 interaction
- `config.py` — local Chroma / hosted Qdrant configuration
- `.env.example` — safe secrets template
- `Dockerfile` — optional Docker deployment

## Memory behaviour

For reliable free-tier usage, one Groq request returns both the chat reply and
a small list of explicit user facts. Those facts are saved with
`infer=False`, so mem0 does not make a second LLM request. The app sends the
saved facts back as context on later turns.
