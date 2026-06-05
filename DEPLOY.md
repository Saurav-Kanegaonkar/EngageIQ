# Deploying EngageIQ to Google Cloud Run

EngageIQ ships as a single container (see [`Dockerfile`](Dockerfile)): a FastAPI app that
serves the SPA and the JSON API, loads the Sentence-Transformers encoder + FAISS index +
cross-encoder re-ranker, renders the topic maps, and produces the WeasyPrint PDF brief.
Cloud Run runs that container behind a public HTTPS URL and scales to zero when idle.

Outcome: a live URL like `https://engageiq-XXXXXXXX-uc.a.run.app` (this is the hosted-URL deliverable).

---

## 0. What you need (one time)

- A **Google Cloud account** with **billing enabled** (Cloud Run has a free monthly allowance;
  scale-to-zero means you pay ~nothing when no one is using it).
- A **project** (note its ID, e.g. `engageiq-prod`).
- The **gcloud CLI** installed and authenticated, *or* just use **Cloud Shell**
  (https://shell.cloud.google.com) which has gcloud + Docker preinstalled and needs no local setup.

> gcloud is **not** installed on this machine. Install it from
> https://cloud.google.com/sdk/docs/install, or run everything below in Cloud Shell
> (open Cloud Shell, `git clone`/upload this folder, then run the same commands).

---

## 1. One-time project setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# the image build (CPU torch + baking two transformer models) can exceed the default
# 10-minute Cloud Build timeout, so raise it:
gcloud config set builds/timeout 1800
```

---

## 2. Deploy (builds the image in the cloud, then deploys)

From the project root (the folder with the `Dockerfile`):

```bash
gcloud run deploy engageiq \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --cpu-boost \
  --timeout 300 \
  --max-instances 1
```

What the flags do:

| Flag | Why |
|---|---|
| `--source .` | Cloud Build builds the `Dockerfile`, pushes the image, and deploys it. No local Docker needed. |
| `--allow-unauthenticated` | Public URL (graders/visitors can open it without a Google login). |
| `--memory 2Gi` | torch + two MiniLM models + the embeddings load to ~1.2-1.8 GB. Bump to `4Gi` if you see an out-of-memory crash. |
| `--cpu 2` + `--cpu-boost` | Faster model load and cross-encoder scoring; boost speeds the cold start. |
| `--timeout 300` | The first request after a cold start waits while the engine warms up. |
| `--max-instances 1` | The per-user learning DB lives in the instance, so one instance keeps everyone's state coherent for a demo. |

When it finishes, gcloud prints the **Service URL** - that is your live link.

---

## 3. (Optional) Enable the live LLM features

The app runs fully **without** an API key: "Suggested actions", card "concepts", and the
plan "read" all fall back to cached/deterministic output for the snapshot. To get fresh
LLM generations for items opened live, add the free NVIDIA key.

Recommended (Secret Manager, key never in your shell history or the image):

```bash
printf "%s" "YOUR_NVIDIA_API_KEY" | gcloud secrets create NVIDIA_API_KEY --data-file=-
# allow the Cloud Run runtime service account to read it:
PROJECT_NUM=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding NVIDIA_API_KEY \
  --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
# re-deploy wiring the secret in as an env var:
gcloud run deploy engageiq --source . --region us-central1 \
  --update-secrets NVIDIA_API_KEY=NVIDIA_API_KEY:latest
```

(Simpler but less private: `--set-env-vars NVIDIA_API_KEY=YOUR_KEY` on the deploy command.
Fine for a class demo; never commit the key.)

---

## 4. Demo-day tips

- **Warm it up first.** Cold start loads the models (~30-60 s). A few minutes before the demo:
  ```bash
  gcloud run services update engageiq --region us-central1 --min-instances 1
  curl -s https://YOUR_URL/api/personas >/dev/null   # trigger the warm-up
  ```
  Set `--min-instances 0` again afterwards to stop paying for the idle instance.
- The **offline snapshot** (corpus + embeddings + cached summaries) is baked into the image,
  so the live app needs no external data sources at request time.

---

## 5. Test the container locally first (optional, needs Docker running)

```bash
docker build -t engageiq:local .
docker run --rm -p 8080:8080 -e PORT=8080 engageiq:local
# open http://localhost:8080
```

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Build times out in Cloud Build | `gcloud config set builds/timeout 1800` (done in step 1). |
| Container crashes / "memory limit exceeded" | Re-deploy with `--memory 4Gi`. |
| First request is slow or times out | Expected cold start; raise `--timeout`, or keep `--min-instances 1` during the demo. |
| Brief PDF font looks different | The image ships `fonts-lmodern` (Latin Modern) so the brief keeps its academic serif. |
| User accounts reset between sessions | The user-state DB is in-instance and ephemeral by design; `--max-instances 1` keeps it stable while warm. |
