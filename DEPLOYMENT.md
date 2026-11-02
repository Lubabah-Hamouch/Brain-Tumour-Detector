# Deployment: Oracle Cloud VM (API) and Vercel (web client)

This deploys the app for free:

- **API:** Docker on an Oracle Cloud "Always Free" virtual machine (Ampere A1, Ubuntu). Two
  containers run there: the API, and [Caddy](https://caddyserver.com) in front of it, which
  serves HTTPS with a free Let's Encrypt certificate that it obtains and renews by itself.
- **Domain:** a free [DuckDNS](https://www.duckdns.org) subdomain, e.g.
  `brain-tumour-api.duckdns.org`, pointing at the VM. Let's Encrypt needs a name, not a bare IP.
- **Web client:** a Vercel project that builds `frontend/`.

The browser loads the site from Vercel and calls the API on the VM directly. Two settings
connect them:

| Where | Setting | Value |
|---|---|---|
| Vercel project | `NEXT_PUBLIC_API_URL` | the API's URL, e.g. `https://brain-tumour-api.duckdns.org` |
| `.env` on the VM | `ALLOWED_ORIGIN` | the Vercel site's URL, e.g. `https://brain-tumour-detector.vercel.app` |

Do the steps in this order: the VM first (sections 1–7), because Vercel needs its URL, then
Vercel (8), then back to the VM to allow the Vercel URL (9).

Credentials stay with you. You sign in to Oracle, DuckDNS and Vercel in the browser, and the keys
you create (the SSH key, and the OCI API key if you use the retry script) stay on your machine. Nothing secret goes into this repository.

## What you need

- An email address and a payment card for the Oracle Cloud sign-up (section 1). Oracle uses the
  card to verify your identity. An Always Free account is not charged.
- A [Vercel](https://vercel.com/signup) account. Sign up with GitHub, so it can see the
  repository.
- A DuckDNS account (sign in with GitHub, Google or another listed provider; nothing to install).
- This repository pushed to GitHub, with the changes you want deployed on `main` (for Vercel).
- On your machine: the project's Python environment (`.venv`) with the sample data in `data/`,
  i.e. what the "try a sample" buttons use locally:
  - `data/raw/brats2021/BraTS2021_00003/`. The bundle script computes the sample study's result
    from it; the files themselves never leave your machine;
  - the cleaned classification test set in `data/processed/figshare_classification_4class_clean/test/`.
- Optional: the Overview's 3D render, `frontend/public/generated/hero-3d.png`
  (`python -m src.visualization.render_hero`).
- For the retry script in step 2.2 (option A): the OCI CLI, set up as described there.
- `ssh` and `scp`. Both come with Windows 10 and 11 (OpenSSH); the commands below are for
  PowerShell.

## Before you start: what becomes public

The API is public: the browser calls it without credentials. Anyone can call its endpoints,
which serve:

- the five sample images from the Brain Tumor MRI Dataset (CC BY 4.0, which allows
  redistribution with attribution; the README gives it);
- the pre-computed sample results, and the hero render if you have one.

**No raw BraTS volumes.** The BraTS terms do not allow redistributing them, so the VM never has
or serves the sample patient's NIfTI files:

- the bundle you copy to the VM does not contain them, and the bundle script stops if any NIfTI
  file ends up in it;
- the API has no `/api/sample-study` or `/api/sample-study/{name}` endpoint there;
- the sample button asks for the pre-computed result and downloads nothing else.

What the API does publish of that patient is derived from the volumes: the predicted label map,
the viewer's slice images (one cropped slice per plane and sequence), the 3D surfaces, the
volumes and the hero render.

The weights are not copied to the VM either. The API container downloads them from the
`weights-v1` GitHub release on its first start.

Only ports 80 and 443 (Caddy) and 22 (SSH) are open to the internet. The API's own port is
reachable only by Caddy, inside Docker.

## 1. Create the Oracle Cloud account

1. Go to https://signup.cloud.oracle.com and sign up for a **Free Tier** account.
2. **Home region:** choose it carefully, because it cannot be changed later and Always Free
   resources live only there. Pick one near you and your visitors (e.g. Germany Central
   (Frankfurt), UK South (London), US East (Ashburn)). Busy regions sometimes run out of free
   Ampere capacity (see "Out of capacity" in [When something is wrong](#when-something-is-wrong)).
3. Enter the card. Oracle may place a small temporary authorisation on it to verify it, which is
   released again. The account stays free: Always Free resources are never charged, and a Free
   Tier account cannot create anything that is not.
4. Wait for the "Your account is ready" email (minutes, sometimes longer), then sign in at
   https://cloud.oracle.com.

## 2. Create the VM

**2.1 Make an SSH key** on your machine (once). In PowerShell:

```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\oracle_vm
```

Press Enter twice for no passphrase, or set one. This creates `oracle_vm` (the private key: stays
on your machine, never share it) and `oracle_vm.pub` (the public key, which goes to Oracle). Show
the public key, to copy it in the next step:

```powershell
Get-Content $HOME\.ssh\oracle_vm.pub
```

Using your own key avoids the Windows file-permission errors that the key Oracle can generate for
you often causes.

**2.2 Create the instance.** Free Ampere capacity is often short: creating the instance can fail
with "Out of host capacity" for hours or days in busy regions. It frees up from time to time, and
the only thing to do is try again. There are two ways to do it:

- **Option A: a script that retries for you** (recommended). It tries every 60–90 seconds until
  it succeeds, and needs the OCI CLI set up once.
- **Option B: the console.** Click through it once. If it reports out of capacity, try again
  later, or switch to option A.

Both create the same instance and need no account change or payment. Then continue with 2.3.

### Option A: create it with the retry script

`deploy/vm/create_instance.py` runs `oci compute instance launch` with the instance's settings. On
"Out of host capacity" it waits 60–90 seconds and tries again, in each of the region's
availability domains in turn. When the instance is running, it prints the instance's details and
public IP. It only ever asks for an Always Free shape (2 OCPUs and 12 GB, or 1 OCPU and 6 GB if
you agree, see A.5). It creates nothing if the compartment already has an Ampere A1 instance, and it
stops on any error other than capacity, rate limiting or a temporary server error.

**A.1 Install the OCI CLI** on your machine. It is a Python package. In PowerShell:

```powershell
python -m pip install --user oci-cli
oci --version
```

If `oci` is not found, pip printed a warning naming the folder it installed it to; add that folder
to your `PATH`. Oracle's own installers are another option:
[Installing the CLI](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm).

**A.2 Create an API signing key** so the CLI can act for you:

1. Collect two OCIDs in the console. Your **user OCID** is under the profile icon (top right),
   then **My profile**, then **OCID** and **Copy**. The **tenancy OCID** is under the profile icon,
   then **Tenancy: …**, then **OCID** and **Copy**. Also note your home region's identifier, e.g.
   `eu-frankfurt-1`. It is shown under **Manage regions** in the region menu.
2. Run `oci setup config`. Give the config location (Enter for the default `~/.oci/config`),
   your user OCID, your tenancy OCID and the region, and answer **Y** to generate a new API
   signing key pair, accepting the default key paths. A passphrase is optional.
3. Upload the public key: profile icon, then **My profile**, then **API keys** (under
   **Resources**, or the **Tokens and keys** tab), then **Add API key**, then **Paste a public
   key**. Paste the contents of `~/.oci/oci_api_key_public.pem`:

   ```powershell
   Get-Content $HOME\.oci\oci_api_key_public.pem
   ```

   Click **Add**.
4. Check that it works: `oci iam availability-domain list` lists your region's availability
   domains. A new key can take a minute to become valid.

The private key (`~/.oci/oci_api_key.pem`) stays on your machine. Treat it like a password: it can
manage your Oracle account.

**A.3 Create the network** (once). The script attaches the instance to an existing public subnet,
which the console otherwise creates along with the instance:

1. In the console menu (☰): **Networking**, then **Virtual cloud networks**, then **Actions** or
   **Start VCN Wizard**.
2. Choose **Create VCN with Internet Connectivity** and click **Start VCN Wizard**.
3. **VCN name:** e.g. `brain-tumour-vcn`. Keep the other defaults, click **Next**, then
   **Create**.
4. Open the new VCN, then **Subnets**, then **public subnet-brain-tumour-vcn** (the *public*
   one), and **Copy** its OCID.

VCNs and subnets are Always Free. If you already have a VCN, e.g. from an earlier attempt with
option B, use its public subnet instead. Section 3 later opens ports 80 and 443 on this
subnet.

**A.4 Fill in the settings.** From the repository root:

```powershell
Copy-Item deploy\vm\instance.env.example deploy\vm\instance.env
notepad deploy\vm\instance.env
```

| Setting | Value |
|---|---|
| `OCI_COMPARTMENT_ID` | the tenancy OCID from A.2: the root compartment, where the free resources go |
| `OCI_SUBNET_ID` | the public subnet's OCID from A.3 |
| `SSH_PUBLIC_KEY_FILE` | the public key from 2.1: `~/.ssh/oracle_vm.pub` |
| `OCI_IMAGE_ID` | leave empty: the script picks the newest **Canonical Ubuntu 24.04** aarch64 image (not "Minimal"). To choose one yourself, take an OCID from `oci compute image list --compartment-id <tenancy OCID> --operating-system "Canonical Ubuntu" --operating-system-version 24.04 --shape VM.Standard.A1.Flex` |
| `OCI_AVAILABILITY_DOMAINS` | leave empty to try all of the region's in turn, or list some, comma-separated |
| `INSTANCE_NAME` | e.g. `brain-tumour-api` |

`instance.env` is gitignored. The same names also work as environment variables, which take
precedence.

**A.5 Run it.** First a dry run, which checks the settings and the CLI and prints the launch
command without launching anything:

```powershell
python deploy/vm/create_instance.py --dry-run
```

Then let it retry:

```powershell
python deploy/vm/create_instance.py
```

Each attempt gets a line:

```
[2026-11-02 22:24:05] Attempt 1: 2 OCPU / 12 GB in xyz:EU-FRANKFURT-1-AD-1
[2026-11-02 22:24:05]   out of host capacity
[2026-11-02 22:24:05]   next attempt in 72 s
```

When the instance is created, the script waits until it is running and prints its name, OCID,
shape, public IP and the `ssh` command. Then continue with 2.3.

- **Stop it** with Ctrl+C. If you stop it during an attempt, check **Compute**, then
  **Instances**; running the script again also reports an instance that was created.
- **Keep the machine awake.** It only retries while it runs, and can take minutes or days. Leave
  the window open and the computer from going to sleep.
- **Rate limits:** if Oracle answers "Too many requests", the script waits longer between
  attempts (up to 5 minutes), then returns to the normal interval.

**The smaller fallback: 1 OCPU and 6 GB.** A smaller instance is sometimes easier to get. After 60
failed attempts at 2 OCPUs and 12 GB (about 75 minutes), the script asks whether to switch. If
you answer no, it keeps trying the full size and does not ask again. It is still Always Free, but:

- **Memory is tight.** At its peak, the app uses about 6.5 GiB with Ubuntu and Caddy (see
  [Sizing](#sizing)), and 6 GB is about 5.6 GiB. The swap file in step 5.4 is **required** on
  this size, not optional. The API's 9 GB memory limit is then above the VM's memory, so swap is
  what keeps a segmentation from exhausting it.
- **Segmentations take about twice as long** on one core.
- **You can resize later:** on the instance's page, **Edit**, then **Edit shape**, set 2 OCPUs
  and 12 GB. This restarts the instance, and it can also fail for lack of capacity, in which case
  the instance keeps its size.

| Option | Effect |
|---|---|
| `--fallback-after N` | offer the smaller size after N failed attempts (default 60; `0` never offers) |
| `--auto-fallback` | switch after N attempts without asking (for an unattended run) |
| `--small` | ask for 1 OCPU and 6 GB from the start |
| `--interval S` | average seconds between attempts (default 75, at least 60) |
| `--max-attempts N` | give up after N attempts (default: keep trying until stopped) |
| `--config PATH` | another settings file instead of `deploy/vm/instance.env` |

### Option B: create it in the console

In the console menu (☰): **Compute**, then **Instances**, then **Create instance**.

1. **Name:** e.g. `brain-tumour-api`.
2. **Image and shape**, click **Edit**:
   - **Image:** click **Change image**, choose **Ubuntu**, then **Canonical Ubuntu 24.04** (the
     plain one, not "Minimal"). Make sure the build is **aarch64** (it is listed for Ampere).
   - **Shape:** click **Change shape**, choose **Ampere**, then **VM.Standard.A1.Flex**.
     Set **Number of OCPUs: 2** and **Amount of memory (GB): 12**. It shows the "Always Free
     eligible" label.

   2 OCPUs and 12 GB is the whole Always Free Ampere allowance (1,500 OCPU hours and 9,000 GB
   hours a month, i.e. 2 OCPUs and 12 GB running all month). It is plenty for this app: see
   [Sizing](#sizing). One Ampere OCPU is one full core, so this is 2 cores.
3. **Networking:** keep **Create new virtual cloud network** and **Create new public subnet**.
   Make sure **Assign a public IPv4 address** is on.
4. **Add SSH keys:** choose **Paste public keys** and paste the line from 2.1
   (`ssh-ed25519 AAAA… `).
5. **Boot volume:** leave the default (about 50 GB). It counts towards the 200 GB of Always Free
   block storage; the app needs about 10 GB of it.
6. Click **Create**. After a minute or two the state changes to **Running**. If it reports
   "Out of host capacity", try another **Availability domain** (under **Placement**), try again
   later, or use option A.

**2.3 Note the public IP.** On the instance's page, under **Instance access**, copy the **Public
IP address**, e.g. `141.147.12.34`. The commands below write it as `<IP>`.

This address stays the instance's for its whole life, including stop and start; only
terminating the instance releases it.

## 3. Open ports 80 and 443 in Oracle's network

Oracle filters traffic **before** it reaches the VM, in the virtual network's security list.
Only SSH (22) is allowed by default. Caddy needs 80 (for Let's Encrypt and the redirect to
HTTPS) and 443 (HTTPS). This is separate from the firewall inside the VM (step 5.3), and both
must allow the ports.

1. On the instance's page, under **Primary VNIC**, click the **Subnet** link
   (`subnet-…`).
2. Open **Security** (or **Security Lists**) and click **Default Security List for vcn-…**.
3. Open **Security rules** (or **Ingress Rules**) and click **Add Ingress Rules**.
4. Fill in:
   - **Source Type:** CIDR. **Source CIDR:** `0.0.0.0/0`.
   - **IP Protocol:** TCP.
   - **Destination Port Range:** `80,443`.
   - **Description:** e.g. `Caddy HTTP and HTTPS`.
5. Click **Add Ingress Rules**. The list now has rules for 22 and for 80,443.

If your subnet uses a Network Security Group instead (**Networking**, then **Network Security
Groups**, attached to the instance's VNIC), add the same rule there. A new instance created as in
step 2 uses the security list.

## 4. Point a DuckDNS subdomain at the VM

1. Go to https://www.duckdns.org and sign in.
2. Under **domains**, type a name, e.g. `brain-tumour-api`, and click **add domain**. If it is
   taken, choose another; the commands below write it as `<name>.duckdns.org`.
3. In the new row, replace **current ip** with the VM's public IP from 2.3 and click **update
   ip**.
4. Check from your machine that the name resolves to the VM (it can take a minute):

```powershell
Resolve-DnsName brain-tumour-api.duckdns.org
```

The answer must show the VM's IP. The IP does not change while the instance exists (2.3), so this
is a one-time setting. Do not share the **token** shown on the DuckDNS page: it lets anyone
change where your name points.

## 5. Prepare the VM

**5.1 Connect over SSH.** From PowerShell, with the IP from 2.3:

```powershell
ssh -i $HOME\.ssh\oracle_vm ubuntu@<IP>
```

Answer `yes` to the fingerprint question the first time. The prompt changes to
`ubuntu@brain-tumour-api:~$`. The commands in 5.2–5.4 and 6.3–6.5 run there, on the VM.

**5.2 Install Docker and Docker Compose** from Docker's official repository:

```bash
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ubuntu
```

The last line lets the `ubuntu` user run Docker without `sudo`. It takes effect at the next
login, so log out (`exit`) and connect again (5.1). Then check:

```bash
docker run --rm hello-world     # prints "Hello from Docker!"
docker compose version          # Docker Compose version v2.x
```

Docker starts on boot by itself, and the stack is set to restart with it, so a reboot of the VM
brings the API back without you doing anything.

**5.3 Open ports 80 and 443 in the VM's firewall.** Oracle's Ubuntu images come with their own
`iptables` rules that reject everything except SSH, independently of the security list in step
3. Allow the two ports, above the reject rule, and save the rules so they survive a reboot:

```bash
sudo iptables -I INPUT -p tcp -m multiport --dports 80,443 -m conntrack --ctstate NEW -j ACCEPT
sudo netfilter-persistent save
sudo iptables -L INPUT -n --line-numbers
```

The list must show the `ACCEPT … multiport dports 80,443` rule **above** the
`REJECT … reject-with icmp-host-prohibited` line. (`-I` inserts at the top; a rule appended with
`-A` would come after the reject and never apply.)

Docker adds its own rules for the ports it publishes, which often let them through even without
this step. Opening them explicitly means HTTPS does not depend on that. Do not enable `ufw` as
well; it would manage the same rules and conflict with Oracle's.

**5.4 Optional: a swap file**, as an extra safety net. The app fits comfortably without one (see
[Sizing](#sizing)):

```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 6. Build the bundle, copy it and start the stack

**6.1 Build the bundle** on your machine. From the repository root, in the project environment:

```powershell
.\.venv\Scripts\Activate.ps1
python deploy/vm/build_bundle.py
```

This writes the bundle to `build/vm/` and the same as one archive, `build/brain-tumour-api.tar.gz`
(both gitignored). It holds:

- the compose stack from `deploy/vm/`: `docker-compose.yml` (the API and Caddy), `Caddyfile`
  and `.env.example`;
- the code and `backend/Dockerfile`, with two settings added for the deployment: no sample-study
  files, and sample results required;
- the five sample images, and the hero render if you have one;
- the sample results. These are computed here, once, with the release weights, which takes a few
  minutes on the CPU. The release weights come from `models/release/` or `models/` when they match,
  and are downloaded into `build/release-weights/` otherwise (about 413 MB).

It ends with `Bundle ready: …; no raw volumes, sample results match the release weights` and the
archive's path.

The sample results are mandatory on the VM, because it has no raw volumes to recompute them from.
Three checks enforce that, so a bad bundle fails loudly instead of serving nothing:

- the bundle script stops if the results are missing or do not match the release weights;
- the image build on the VM runs the same check, and fails;
- the API refuses to start without a matching file.

**6.2 Copy the archive to the VM.** From the repository root, in PowerShell on your machine:

```powershell
scp -i $HOME\.ssh\oracle_vm build\brain-tumour-api.tar.gz ubuntu@<IP>:~
```

**6.3 Unpack it** on the VM (over SSH, 5.1):

```bash
tar -xzf brain-tumour-api.tar.gz
cd brain-tumour-api
```

**6.4 Set the domain and the allowed origin.** Create `.env` from the example:

```bash
cp .env.example .env
nano .env
```

Set `DOMAIN` to your DuckDNS name from step 4, e.g. `DOMAIN=brain-tumour-api.duckdns.org`.
Leave `ALLOWED_ORIGIN` as it is for now; section 9 sets it once the Vercel site exists. Save with
Ctrl+O, Enter, and exit with Ctrl+X.

**6.5 Start the stack:**

```bash
docker compose up -d --build
```

- The first build takes about 5–10 minutes: it installs PyTorch for ARM and checks the sample
  results (the `RUN python -m backend.services.samples` step). Later builds reuse what they can.
- `-d` runs the stack in the background; it keeps running after you log out.

Follow the start-up:

```bash
docker compose logs -f
```

Press Ctrl+C to stop following (the stack keeps running). A healthy start shows:

- `api`: the weights download (`models/segmentation/fold1/best.pt <- segmentation_fold1.pt` …),
  then `loaded pre-computed sample results` and `Application startup complete`;
- `caddy`: `certificate obtained successfully` for your domain, within a minute of starting.

`docker compose ps` shows both services; the API turns `healthy` once its models are loaded.

## 7. Check HTTPS and the API

From your machine, open `https://<name>.duckdns.org/api/health` in the browser. It must load
with a padlock and no certificate warning, and show `"status":"ok"`, `"device":"cpu"` and
`"sample_results":"ready"`. The interactive API docs are at `https://<name>.duckdns.org/docs`.

Or in PowerShell:

```powershell
curl.exe -s https://brain-tumour-api.duckdns.org/api/health
curl.exe -s -o NUL -w "%{http_code} %{redirect_url}\n" http://brain-tumour-api.duckdns.org/api/health
```

The second line checks the redirect: it prints `308 https://brain-tumour-api.duckdns.org/api/health`.

If the certificate does not arrive, see [When something is wrong](#when-something-is-wrong):
nearly always DNS (step 4) or a closed port (steps 3 and 5.3). Caddy retries by itself, so after
fixing the cause, wait a minute and check `docker compose logs caddy` again.

`https://<name>.duckdns.org` is the API's URL for the next steps.

## 8. Create the Vercel project (the web client)

1. On vercel.com, click **Add New…**, then **Project**.
2. Next to `Brain-Tumour-Detector`, click **Import**. If the repository is not listed, click
   **Adjust GitHub App Permissions** and give Vercel access to it.
3. On the **Configure Project** screen:
   - **Root Directory:** click **Edit**, choose **`frontend`** and click **Continue**. This
     matters: the app is in `frontend/`, not the repository root.
   - **Framework Preset:** Next.js (detected). Leave the build and output settings as they are.
   - **Environment Variables:** add
     - **Key** `NEXT_PUBLIC_API_URL`, **Value** your API URL from step 7, e.g.
       `https://brain-tumour-api.duckdns.org`, with no trailing slash;
     - optional, for the Overview's 3D image: **Key** `HERO_IMAGE_URL`, **Value** the API URL
       followed by `/api/hero.png`. This only works if the bundle has the render (step 6.1 says
       if it has not). Without it, the Overview shows a note where the image would be.
4. Click **Deploy**. The build takes 1–2 minutes.
5. When it finishes, note the site's address. It is under **Domains** on the project page, e.g.
   `https://brain-tumour-detector.vercel.app`. Use this production domain, not the
   longer per-deployment URL.

If the build fails with `NEXT_PUBLIC_API_URL is not set`, add the variable as in step 3:
**Settings**, then **Environment Variables**. Then redeploy: **Deployments**, then **⋯** on the
latest deployment, then **Redeploy**.

`NEXT_PUBLIC_API_URL` is fixed into the site when it is built. After changing it, redeploy;
saving the variable alone changes nothing.

## 9. Allow the Vercel site to call the API

The API answers browser requests only from the origins it allows. By default those are local
development addresses. On the VM, in `~/brain-tumour-api`:

```bash
nano .env        # ALLOWED_ORIGIN=https://brain-tumour-detector.vercel.app
docker compose up -d
```

- Use your address from step 8.5: `https://`, and no trailing slash or path.
- For more than one address, e.g. a custom domain as well, separate them with commas:
  `ALLOWED_ORIGIN=https://a.vercel.app,https://www.example.com`.
- `docker compose up -d` recreates the API container with the new setting. A new container
  downloads the weights again, so the API is back after about a minute. Caddy keeps running and
  keeps its certificate.

## 10. Check that they talk to each other

**From the command line** (with your two addresses):

```powershell
curl.exe -s -D - -o NUL -H "Origin: https://brain-tumour-detector.vercel.app" https://brain-tumour-api.duckdns.org/api/health
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
   `https://` requests to your `duckdns.org` name.

Your own uploads run the models on the VM's 2 cores. A classification takes a few seconds. A
segmentation takes roughly 15–30 minutes, with live progress (see [Sizing](#sizing)).

## Always on: no cold starts

The VM runs all the time, and so does the API. It downloads the weights once, when its container
is first created, and keeps them across restarts and reboots. So unlike a scale-to-zero host
(such as Google Cloud Run, where each new instance starts from scratch), there is no cold start: no weights download
and no model loading when a visitor arrives after a quiet spell. Every request is answered at
normal speed, and the sample buttons answer at once.

The weights are downloaded again only when the API container is recreated: after an update
(section "Updating later") or a change to `.env`. That takes about a minute, inside Oracle's
network.

## Sizing

**Memory: the app needs at most about 6 GiB; the VM has 12 GB.** Measured on this deployment
(the bundle's image and compose stack, requests going through Caddy over HTTPS), with the API
container limited to 2 CPU cores as on the VM. The test ran three segmentations of different
BraTS studies back to back, with two classifications every 10 seconds alongside, and sampled the
memory every 2 seconds for 65 minutes:

| Situation | Memory |
|---|---|
| API idle after a fresh start, all six models loaded | 1.2–1.6 GiB |
| API during a segmentation, with classifications alongside | **5.6–5.9 GiB** (peak, in each of the three) |
| API idle after each of the three segmentations | 1.5, 2.3, 1.9 GiB |
| Caddy | 20–40 MiB |

- **No growth over time.** After a job the API does not drop all the way back to its fresh-start
  level: it keeps the recent results for download, and the memory allocator keeps some freed
  memory for reuse. But that does not build up from job to job (1.5, 2.3, 1.9 GiB above), and
  every segmentation peaks at the same level. A long-running API behaves like a fresh one.
- **Headroom:** at the peak, the API (about 6 GiB), Caddy and Ubuntu itself (about 0.5 GiB)
  together use about 6.5 GiB of the VM's 12 GB (about 11.2 GiB usable), leaving about 4.5 GiB
  free. No swap is needed.
- **Measured on x86, not on the Ampere VM.** PyTorch on ARM uses different kernels, so the
  numbers there will differ somewhat, but not by gigabytes. Check them on the VM with
  `docker stats` during your first segmentation.

- **Jobs run one at a time.** Segmentations queue rather than run in parallel, so the peak does
  not grow with the number of visitors; more visitors only make the queue longer.
- **Results kept in memory are bounded:** the 16 most recent results (`BT_RESULT_CACHE_SIZE`),
  then the oldest are dropped.
- **A memory limit of 9 GB** is set on the API container (`mem_limit` in `docker-compose.yml`),
  well above its peak. If something ever went wrong and it grew past that, Docker would restart
  the container instead of the whole VM running out of memory and becoming unreachable over SSH.

**CPU: 2 cores.** A classification takes a few seconds. A segmentation runs five models over the
whole volume: 14–32 minutes in the test above, on 2 x86 cores with classifications alongside.
Ampere cores are full cores (no hyper-threading), so expect the same order of time; it has not
been measured on Ampere. While a segmentation runs, it keeps both cores busy. The site stays
responsive (Caddy retries a connection to the API that stalls, see `deploy/vm/Caddyfile`), and SSH
still works.

**Disk:** the image is about 2 GB and the weights 0.4 GB; the 50 GB boot volume has plenty of
room. Docker's logs are capped at 30 MB per container (`logging` in `docker-compose.yml`), so a
long-running VM does not fill its disk with them.

## Staying within Always Free

- **Compute:** 2 OCPUs and 12 GB of Ampere A1 is the whole allowance, so do not create a second
  A1 instance alongside this one.
- **Storage:** 200 GB of block storage in total (boot volumes included); this VM uses 50 GB.
- **Traffic:** 10 TB a month of outbound data is free, far more than this app's small images and
  JSON responses need.
- **Idle instances can be reclaimed.** Oracle may stop an Always Free instance it deems idle:
  one where, over 7 days, CPU use (95th percentile), network use and memory use all stay below
  20%. An API with few visitors is idle by that measure: CPU and network are near zero, and
  memory is about 15–20% of 12 GB when nothing runs. So:
  - check on the instance now and then (**Compute**, then **Instances**). If it was stopped,
    click **Start**: the stack comes back by itself, with the same IP and certificate;
  - or upgrade the account to **Pay As You Go** (**Billing & Cost Management**, then **Upgrade
    and Manage Payment**). Always Free resources stay free after the upgrade, and the idle
    reclamation is widely reported not to apply to such accounts (Oracle's page does not say so
    explicitly). If you upgrade, add a budget with an alert at a small amount (**Billing & Cost
    Management**, then **Budgets**), since a paid account could then create resources that are
    not free.

## When something is wrong

| Symptom | Cause and fix |
|---|---|
| Creating the instance: "Out of capacity for shape VM.Standard.A1.Flex" | The region has no free Ampere capacity at the moment; it comes and goes. Let the script retry for you (step 2.2, option A), or in the console try another **Availability domain** (under **Placement**) and try again later. |
| `create_instance.py`: `NotAuthenticated` or "config file … is invalid" | The API key is not set up or not uploaded yet (step 2.2, A.2). `oci iam availability-domain list` must work first. |
| `create_instance.py`: `NotAuthorizedOrNotFound` or `InvalidParameter` | An OCID in `deploy/vm/instance.env` is wrong or from another region: check the compartment (the tenancy OCID) and the subnet (A.3, A.4). |
| `ssh`: "Connection timed out" | Wrong IP, or the instance is not running. Port 22 is open by default; if you edited the security list, check that its rule for 22 is still there. |
| `ssh`: "Permission denied (publickey)" | Use `-i $HOME\.ssh\oracle_vm` and the user `ubuntu`. The key must be the one pasted in step 2.2. |
| `docker: permission denied … docker.sock` | You have not logged in again since adding `ubuntu` to the `docker` group (5.2). `exit` and connect again. |
| `docker compose up`: "set DOMAIN in .env" | `.env` is missing or has no `DOMAIN` (6.4). Run the command from `~/brain-tumour-api`, where `.env` is. |
| `caddy` logs: "timeout during connect", "connection refused" or "Fetching http://…/.well-known/acme-challenge" errors | Let's Encrypt cannot reach the VM on port 80. Check the security list rule (step 3) and the VM's firewall (5.3): `curl http://<name>.duckdns.org` from your machine must answer (a redirect is fine). |
| `caddy` logs: "NXDOMAIN" or "no valid A records" | The DuckDNS name does not point at the VM yet (step 4). Fix it; Caddy retries by itself. |
| `caddy` logs: "too many certificates" or "rateLimited" | Let's Encrypt limits how often one name is issued. Wait as the message says. Avoid `docker compose down -v`, which deletes the stored certificate (the `caddy_data` volume) and forces a new one. |
| Browser: the page loads, but the API answers 502 | Caddy is up, the API is not (yet). `docker compose logs api`: it may still be downloading the weights or loading the models. |
| Build fails at `python -m backend.services.samples`, or the API logs show `BT_SAMPLE_RESULTS_REQUIRED` | The sample results are missing or were computed for other weights or settings. Rebuild the bundle (6.1) and deploy it again ("Updating later"). |
| API logs: `Download failed`, or a GitHub error | The weights could not be downloaded. Check that the repository's `weights-v1` release is public, then `docker compose restart api`. |
| Vercel build: `NEXT_PUBLIC_API_URL is not set` | Add the variable (step 8.3) and redeploy. |
| Site: "The API at … is not reachable", or the Overview says the results are unavailable | `curl.exe -s https://<name>.duckdns.org/api/health` from your machine. If that fails, check `docker compose ps` on the VM, and that the instance is running (it may have been reclaimed; see [Staying within Always Free](#staying-within-always-free)). |
| Console: "blocked by CORS policy" | `ALLOWED_ORIGIN` does not equal the site's address exactly (scheme, no trailing slash). Fix it (section 9). Preview deployments have their own addresses; test on the production domain or add them. |
| `docker compose ps` shows the API restarting, `docker inspect` says `OOMKilled` | It went past its 9 GB limit, which it should not do (see [Sizing](#sizing)). Look at `docker compose logs api` for what it was doing. |

## Updating later

- **Web client:** push to `main` on GitHub. Vercel builds and deploys each push automatically.
- **API:** rebuild the bundle (6.1) and copy it (6.2). Then on the VM:

```bash
rm -rf ~/brain-tumour-api.new && mkdir ~/brain-tumour-api.new
tar -xzf ~/brain-tumour-api.tar.gz -C ~/brain-tumour-api.new --strip-components=1
cp ~/brain-tumour-api/.env ~/brain-tumour-api.new/
rm -rf ~/brain-tumour-api && mv ~/brain-tumour-api.new ~/brain-tumour-api
cd ~/brain-tumour-api && docker compose up -d --build
```

  This replaces the folder (so files removed from the code do not linger) and keeps your `.env`.
  Caddy's certificate lives in a Docker volume and is kept. The API is unavailable for the few
  minutes of the build's last steps and the weights download. Afterwards, `docker image prune -f`
  removes the old image.
- **Logs:** `docker compose logs --tail 100 api` (or `caddy`), in `~/brain-tumour-api`.
- **Restart:** `docker compose restart api`. **Stop:** `docker compose down` (keeps the
  certificate; do not add `-v`).
- **Taking it offline:** `docker compose down` on the VM stops the API. To remove everything,
  terminate the instance in the console (**Compute**, **Instances**, **⋯**, **Terminate**, with
  **Permanently delete the attached boot volume** ticked), and delete the DuckDNS subdomain.
