# Deployment

RetinaScan AI runs on GCP Cloud Run with a Supabase (Postgres) backend.

## Architecture
- **App**: Streamlit, containerized via `deploy/Dockerfile`
- **Database**: Supabase Postgres (Session Pooler connection — required for IPv4 networks)
- **File storage**: Supabase Storage (`gradcam-images` bucket) for Grad-CAM composite images
- **Compute**: Google Cloud Run (region: asia-south1), scales to zero when idle
- **CI**: Cloud Build (`deploy/cloudbuild.yaml`) — pulls the model from a GCS bucket (kept out of the git-based build context) before the Docker build

## Redeploying
The container image lives in Artifact Registry, so most redeploys don't need a rebuild:
```bash
gcloud run deploy retinascan-ai \
    --image=asia-south1-docker.pkg.dev/retinascan-ai-507111/retinascan-repo/retinascan-ai \
    --platform=managed --region=asia-south1 \
    --memory=4Gi --cpu=2 --min-instances=0 --max-instances=3 \
    --allow-unauthenticated \
    --set-env-vars="DB_BACKEND=supabase,SUPABASE_DB_HOST=...,SUPABASE_DB_PORT=5432,SUPABASE_DB_NAME=postgres,SUPABASE_DB_USER=...,SUPABASE_DB_PASSWORD=...,SUPABASE_URL=...,SUPABASE_SERVICE_ROLE_KEY=..."
```

## Taking the service down (cost control)
```bash
gcloud run services delete retinascan-ai --region=asia-south1
```
The image stays in Artifact Registry — redeploy anytime with the command above.

## Rebuilding after a code change
```bash
gcloud builds submit --config=deploy/cloudbuild.yaml .
```