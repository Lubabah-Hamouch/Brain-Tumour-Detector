# Deployment: Hugging Face Space (API) and Vercel (web client)

This deploys the app on free tiers:

- **API:** a Docker Space on Hugging Face, on the free CPU hardware (2 vCPU, 16 GB RAM). It
  builds `backend/Dockerfile`.
- **Web client:** a Vercel project that builds `frontend/`.

The browser loads the site from Vercel and calls the API on the Space directly. Two settings
connect them:

| Where | Setting | Value |
|---|---|---|
| Vercel project | `NEXT_PUBLIC_API_URL` | the Space's URL, e.g. `https://your-name-brain-tumour-api.hf.space` |
| Hugging Face Space | `ALLOWED_ORIGIN` | the Vercel site's URL, e.g. `https://brain-tumour-detector.vercel.app` |

Do the steps in this order: the Space first, because Vercel needs its URL, then Vercel, then go
back to the Space to allow the Vercel URL.

Tokens and keys stay on your machine. You type them into the Hugging Face CLI login and the two
websites, never into a file in this repository.

## What you need

- A [Hugging Face](https://huggingface.co/join) account and a [Vercel](https://vercel.com/signup)
  account. For Vercel, sign up with GitHub, so it can see the repository.
- This repository pushed to GitHub, with the changes you want deployed on `main`.
- On your machine: the project's Python environment (`.venv`) with the sample data in `data/`,
  i.e. what the "try a sample" buttons use locally:
  - `data/raw/brats2021/BraTS2021_00003/`. The bundle script computes the sample study's result
    from it; the files themselves are never uploaded;
  - the cleaned classification test set in `data/processed/figshare_classification_4class_clean/test/`.
- Optional: the Overview's 3D render, `frontend/public/generated/hero-3d.png`
  (`python -m src.visualization.render_hero`).

## Before you start: what becomes public

A free Space has to be **public**, because the browser calls it without a token. Everything
uploaded to it can be browsed and downloaded by anyone at `huggingface.co/spaces/<you>/<space>`:

- the code;
- the five sample images from the Brain Tumor MRI Dataset (CC BY 4.0, which allows
  redistribution with attribution; the Space's card gives it);
- the pre-computed sample results, and the hero render if you have one.

**No raw BraTS volumes.** The BraTS terms do not allow redistributing them, so neither the Space
nor its API offers the sample patient's NIfTI files:

- the bundle does not contain them, and the bundle script stops if any NIfTI file ends up in it;
- the Space's API has no `/api/sample-study` or `/api/sample-study/{name}` endpoint;
- the sample button asks for the pre-computed result and downloads nothing else.

What the Space does publish of that patient is derived from the volumes: the predicted label map,
the viewer's slice images (one cropped slice per plane and sequence), the 3D surfaces, the
volumes and the hero render.

The weights are not uploaded. The Space downloads them from the `weights-v1` GitHub release when
it starts.

## 1. Create the Hugging Face Space (the API)

**1.1 Get a write token.**

1. On huggingface.co, click your avatar (top right), then **Settings**, then **Access Tokens**.
2. Click **Create new token**, choose the **Write** type and give it a name, e.g. `space-upload`.
3. Copy the token. You paste it once in step 1.3.

**1.2 Create the Space.**

1. Go to https://huggingface.co/new-space.
2. Fill in the form:
   - **Owner:** your user name. **Space name:** e.g. `brain-tumour-api`.
   - **License:** anything. The card uploaded in step 1.5 replaces it.
   - **Select the Space SDK:** **Docker**, then the **Blank** template.
   - **Space hardware:** **CPU basic · 2 vCPU · 16 GB · FREE**.
   - **Visibility:** **Public**.
3. Click **Create Space**.
4. Note the Space's address. The API's URL is `https://<owner>-<space-name>.hf.space`,
   lower-case, e.g. `https://your-name-brain-tumour-api.hf.space`. It is also shown in the
   Space's **⋮** menu, under **Embed this Space**, as "Direct URL".

**1.3 Log in from your machine** (once per machine):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -U huggingface_hub
hf auth login          # paste the token from 1.1; answer "n" to adding it as a git credential
```

The token is saved in your user profile (`~/.cache/huggingface/`), not in the project.

**1.4 Build the Space's files.** From the repository root:

```powershell
python deploy/huggingface/build_space.py
```

This writes the Space's contents to `build/hf-space/` (gitignored):

- the code and `backend/Dockerfile`, with two settings added for the Space: no sample-study
  files, and sample results required;
- the Space card `deploy/huggingface/README.md`, which tells Hugging Face this is a Docker Space
  on port 8010;
- the five sample images;
- the sample results. These are computed here, once, with the release weights, which takes a few
  minutes on the CPU. The release weights come from `models/release/` or `models/` when they match,
  and are downloaded into `build/release-weights/` otherwise (about 410 MB).

It ends with `Space bundle ready; no raw volumes, sample results match the release weights`.

The sample results are mandatory on the Space, because it has no raw volumes to recompute them
from. Three checks enforce that, so a bad bundle fails loudly instead of serving nothing:

- the bundle script stops if the results are missing or do not match the release weights;
- the Space's image build runs the same check, and fails;
- the API refuses to start without a matching file.

**1.5 Upload them** (replace `your-name/brain-tumour-api` with your Space):

```powershell
hf upload your-name/brain-tumour-api build/hf-space . --repo-type space
```

**1.6 Watch it start.**

1. Open the Space page and click **Logs**.
2. **Build** shows the Docker build: about 5–10 minutes the first time.
3. **Container** shows the start-up:
   - the weights download (`models/segmentation/fold1/best.pt <- segmentation_fold1.pt` …,
     about 410 MB);
   - `loaded pre-computed sample results`;
   - `Application startup complete`.
4. The status badge turns **Running**.

**1.7 Check the API.** Open `https://<owner>-<space-name>.hf.space/api/health`. You should see
`"status":"ok"`, `"device":"cpu"` and `"sample_results":"ready"`. The interactive API docs are
at `/docs` on the same address.

## 2. Create the Vercel project (the web client)

1. On vercel.com, click **Add New…**, then **Project**.
2. Next to `Brain-Tumour-Detector`, click **Import**. If the repository is not listed, click
   **Adjust GitHub App Permissions** and give Vercel access to it.
3. On the **Configure Project** screen:
   - **Root Directory:** click **Edit**, choose **`frontend`** and click **Continue**. This
     matters: the app is in `frontend/`, not the repository root.
   - **Framework Preset:** Next.js (detected). Leave the build and output settings as they are.
   - **Environment Variables:** add
     - **Key** `NEXT_PUBLIC_API_URL`, **Value** your Space's URL from step 1.2, e.g.
       `https://your-name-brain-tumour-api.hf.space`, with no trailing slash;
     - optional, for the Overview's 3D image: **Key** `HERO_IMAGE_URL`, **Value**
       `https://huggingface.co/spaces/<owner>/<space-name>/resolve/main/assets/hero-3d.png`.
       This only works if the bundle has `assets/hero-3d.png` (step 1.4 says if it has not).
       Without it, the Overview shows a note where the image would be.
4. Click **Deploy**. The build takes 1–2 minutes.
5. When it finishes, note the site's address. It is under **Domains** on the project page, e.g.
   `https://brain-tumour-detector.vercel.app`. Use this production domain, not the
   longer per-deployment URL.

If the build fails with `NEXT_PUBLIC_API_URL is not set`, add the variable as in step 3:
**Settings**, then **Environment Variables**. Then redeploy: **Deployments**, then **⋯** on the
latest deployment, then **Redeploy**.

`NEXT_PUBLIC_API_URL` is fixed into the site when it is built. After changing it, redeploy;
saving the variable alone changes nothing.

## 3. Allow the Vercel site to call the API

The API answers browser requests only from the origins it allows. By default those are local
development addresses.

1. On the Space page, open **Settings**, then **Variables and secrets**.
2. Click **New variable** (a variable, not a secret; the value is not confidential):
   - **Name** `ALLOWED_ORIGIN`;
   - **Value** the Vercel address from step 2.5, e.g. `https://brain-tumour-detector.vercel.app`.
     Use `https://` and no trailing slash or path.
3. Save. The Space restarts with the new setting, which takes a few minutes. Watch **Logs**
   again until it is **Running**.

For more than one address, e.g. a custom domain as well, separate them with commas.

## 4. Check that they talk to each other

**From the command line:**

```powershell
curl.exe -s -D - -o NUL -H "Origin: https://brain-tumour-detector.vercel.app" https://your-name-brain-tumour-api.hf.space/api/health
```

The headers must include
`access-control-allow-origin: https://brain-tumour-detector.vercel.app`. If that line is missing,
`ALLOWED_ORIGIN` does not match the address exactly.

**In the browser,** open the Vercel site with the developer tools open (F12, **Console** tab):

1. **Overview:** the headline results (Dice, accuracy) appear. The Vercel server fetches these,
   so this proves the URL is right, but not yet CORS.
2. **Classify:** the five sample images are listed. Click one: the result appears within a
   second or two.
3. **Segment:** click **Run the sample patient**. The result opens almost at once, without any
   file being downloaded or uploaded. The caption says "pre-computed".
4. In the result, open the **3D** view: the brain and tumour surfaces load.
5. The console shows no errors mentioning CORS or "blocked". The **Network** tab shows only
   `https://` requests to `hf.space`.

Your own uploads still run the models on the Space's CPU. A classification takes a few seconds.
A segmentation takes several minutes, with live progress.

## When something is wrong

| Symptom | Cause and fix |
|---|---|
| Vercel build: `NEXT_PUBLIC_API_URL is not set` | Add the variable (step 2.3) and redeploy. |
| Site: "The API at … is not reachable", or the Overview says the results are unavailable | The Space is asleep or starting. Open the Space page to wake it, wait until **Running**, reload. |
| Console: "blocked by CORS policy" | `ALLOWED_ORIGIN` on the Space does not equal the site's address exactly (scheme, no trailing slash). Fix it (step 3). Preview deployments have their own addresses; test on the production domain or add them. |
| Space logs: `Download failed`, or a GitHub error | The weights could not be downloaded. Check that the repository's `weights-v1` release is public, then restart the Space: **Settings**, **Factory reboot**. |
| Space build fails at `python -m backend.services.samples`, or the container stops with `BT_SAMPLE_RESULTS_REQUIRED` | The sample results are missing or were computed for other weights or settings. Rebuild the bundle (step 1.4) and upload it again (step 1.5). |
| Space build fails | Open **Logs**, **Build**. Rebuild the bundle (step 1.4) and upload it again (step 1.5). |

## Updating later

- **Web client:** push to `main` on GitHub. Vercel builds and deploys each push automatically.
- **API:** rebuild the bundle and upload it again (steps 1.4 and 1.5). The Space rebuilds on
  every upload.
- **Free-tier behaviour:** a free Space sleeps after 48 hours without visitors. The next visit
  wakes it, and the restart downloads the weights again, which takes a few minutes. It has no
  persistent disk: anything written at run time is lost on restart. The bundled sample results
  are part of the image, so they survive.
