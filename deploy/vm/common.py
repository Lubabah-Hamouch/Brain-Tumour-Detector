"""Helpers shared by the deploy/vm scripts: settings, the OCI CLI, SSH, and checks of the API.

Not run on its own. The scripts that use it (create_network.py, create_instance.py,
setup_remote.py, update_origin.py) import it from this folder, and use only the standard library
and the `oci`, `ssh`, `scp` and `ssh-keygen` commands.
"""

import configparser
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SETTINGS_FILE = HERE / "instance.env"
SETTINGS_EXAMPLE = HERE / "instance.env.example"
DEFAULT_SSH_KEY = "~/.ssh/oracle_vm.pub"

REMOTE_DIR = "~/brain-tumour-api"         # the stack's folder on the VM (DEPLOYMENT.md, section 6)
ARCHIVE = ROOT / "build" / "brain-tumour-api.tar.gz"


# ---- Output ---------------------------------------------------------------------------------

def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def fail(message):
    sys.exit(f"Error: {message}")


# ---- Settings (deploy/vm/instance.env) ----------------------------------------------------

SETTING_KEYS = ["OCI_COMPARTMENT_ID", "OCI_SUBNET_ID", "SSH_PUBLIC_KEY_FILE", "OCI_IMAGE_ID",
                "OCI_AVAILABILITY_DOMAINS", "INSTANCE_NAME", "INSTANCE_IP", "DOMAIN"]


def is_set(value):
    """Empty values and the example's placeholders (ocid1.tenancy.oc1...) count as not set."""
    return bool(value) and "..." not in value


def read_settings(path=SETTINGS_FILE):
    """KEY=value lines from the settings file, overridden by environment variables."""
    settings = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip().strip('"').strip("'")
    for key in SETTING_KEYS:
        if os.environ.get(key):
            settings[key] = os.environ[key].strip()
    return {key: value for key, value in settings.items() if is_set(value)}


def write_settings(updates, path=SETTINGS_FILE):
    """Set KEY=value lines in the settings file, created from the example if it does not exist.

    Other lines and comments are kept; a key that is not in the file yet is appended.
    """
    if not path.exists():
        shutil.copyfile(SETTINGS_EXAMPLE, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    for key, value in updates.items():
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"{key}={value}"
                break
        else:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def oci_config_tenancy():
    """The tenancy OCID from the OCI CLI's config file (the profile in OCI_CLI_PROFILE, or DEFAULT)."""
    path = Path(os.path.expanduser(os.environ.get("OCI_CLI_CONFIG_FILE", "~/.oci/config")))
    if not path.exists():
        return None
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    profile = os.environ.get("OCI_CLI_PROFILE", "DEFAULT")
    section = config.defaults() if profile == "DEFAULT" else (config[profile] if profile in config else {})
    return section.get("tenancy")


# ---- The OCI CLI ----------------------------------------------------------------------------

class OciError(Exception):
    """An OCI CLI call failed. kind: capacity, rate_limit, transient or fatal."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def classify(stderr):
    """Sort a failed CLI call into what to do next, from the ServiceError it prints."""
    status, code, message = None, "", stderr.strip()
    start = stderr.find("{")
    if start != -1:
        try:
            error = json.loads(stderr[start:])
            status, code = error.get("status"), str(error.get("code", ""))
            message = str(error.get("message", message))
        except ValueError:
            pass
    text = f"{code} {message}".lower()
    if "out of host capacity" in text or "out of capacity" in text:
        return OciError("capacity", message)
    if status == 429 or "toomanyrequests" in text:
        return OciError("rate_limit", message)
    if status in (502, 503, 504) or "timed out" in text or "connection" in text:
        return OciError("transient", message)
    return OciError("fatal", stderr.strip() or "the OCI CLI failed without a message")


def oci(*args):
    """Run the OCI CLI and return its JSON output (None if it printed nothing)."""
    exe = shutil.which("oci")
    if exe is None:
        fail("the OCI CLI (`oci`) is not on PATH; DEPLOYMENT.md, section 2, explains how to install it.")
    result = subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise classify(result.stderr)
    out = result.stdout.strip()
    # With --wait-for-state, the CLI can print progress before the resource's JSON.
    return json.loads(out[out.index("{"):]) if "{" in out else None


# ---- Inputs ---------------------------------------------------------------------------------

def check_ip(value):
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        fail(f"not an IP address: {value!r}")


def check_domain(value):
    domain = value.strip().lower().rstrip(".")
    domain = re.sub(r"^https?://", "", domain).rstrip("/")
    label = r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?"
    if not re.fullmatch(rf"({label}\.)+[a-z]{{2,63}}", domain) or len(domain) > 253:
        fail(f"not a host name: {value!r} (expected e.g. brain-tumour-api.duckdns.org)")
    return domain


def check_origins(value):
    """Comma-separated https:// origins without path; a trailing slash is dropped."""
    origins = []
    for item in value.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if not re.fullmatch(r"https://[a-z0-9]([a-z0-9.-]*[a-z0-9])?(:\d{1,5})?", origin.lower()):
            fail(f"not an https:// origin: {item.strip()!r} "
                 "(expected e.g. https://brain-tumour-detector.vercel.app, with no path)")
        origins.append(origin.lower())
    return ",".join(origins)


# ---- SSH ------------------------------------------------------------------------------------

def ssh_private_key(settings):
    """The private key that belongs to SSH_PUBLIC_KEY_FILE (the same path without .pub)."""
    public = Path(os.path.expanduser(settings.get("SSH_PUBLIC_KEY_FILE", DEFAULT_SSH_KEY)))
    private = public.with_suffix("") if public.suffix == ".pub" else public
    if not private.is_file():
        fail(f"SSH private key not found: {private}. It is the SSH_PUBLIC_KEY_FILE in "
             f"{SETTINGS_FILE.name} without .pub; create_network.py creates the pair.")
    return private


class Remote:
    """Runs commands on the VM over SSH as `user`, with the deployment's key."""

    def __init__(self, ip, key, user="ubuntu"):
        self.target = f"{user}@{ip}"
        self.ssh, self.scp = shutil.which("ssh"), shutil.which("scp")
        if self.ssh is None or self.scp is None:
            fail("ssh and scp are not on PATH (on Windows: Settings > Apps > Optional features > OpenSSH Client).")
        # accept-new trusts the VM's host key the first time, as answering "yes" would, and refuses
        # a changed one afterwards. BatchMode fails instead of prompting for a password.
        self.options = ["-i", str(key), "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]

    def run(self, script, capture=False, check=True):
        """Run a bash script on the VM, passed on stdin. Output streams to the console unless captured."""
        # The braces make bash read the whole script before running it, so a command that reads
        # stdin cannot swallow the lines after it. Bytes, not text: on Windows, text mode would
        # turn \n into \r\n, which bash rejects.
        result = subprocess.run([self.ssh, *self.options, self.target, "bash -s"],
                                input=("{\n" + script + "\n}\n").encode("utf-8"),
                                stdout=subprocess.PIPE if capture else None,
                                stderr=subprocess.PIPE if capture else None)
        output = result.stdout.decode("utf-8", "replace") if capture else ""
        if check and result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip() if capture else ""
            raise RuntimeError(f"the command on the VM failed (exit {result.returncode}){': ' + detail if detail else ''}")
        return output if capture else result.returncode

    def write_file(self, remote_path, content):
        """Write a text file on the VM, with LF line endings."""
        subprocess.run([self.ssh, *self.options, self.target, f"cat > {remote_path}"],
                       input=content.replace("\r\n", "\n").encode("utf-8"), check=True)

    def upload(self, local_path, remote_path):
        subprocess.run([self.scp, *self.options, str(local_path), f"{self.target}:{remote_path}"], check=True)

    def wait_until_reachable(self, timeout=600):
        """A new instance takes a minute or two to accept SSH; retry until then."""
        deadline = time.time() + timeout
        while True:
            result = subprocess.run([self.ssh, *self.options, self.target, "true"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if result.returncode == 0:
                return
            error = result.stderr.decode("utf-8", "replace").strip()
            if "Permission denied" in error or "Host key verification failed" in error:
                raise RuntimeError(error)
            if time.time() > deadline:
                raise RuntimeError(f"no SSH connection after {timeout // 60} minutes: {error}")
            log(f"  not reachable yet ({error.splitlines()[-1] if error else 'no answer'}); retrying in 15 s")
            time.sleep(15)


def parse_env(text):
    values = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


# ---- Checks of the running API ----------------------------------------------------------------

API_HEALTH = """
cd {dir}
id=$(docker compose ps -q api)
[ -n "$id" ] || {{ echo missing; exit 0; }}
docker inspect --format '{{{{if .State.Health}}}}{{{{.State.Health.Status}}}}{{{{else}}}}{{{{.State.Status}}}}{{{{end}}}}' "$id"
"""

# Runs in the API container, which has Python but not curl.
API_REQUEST = """
cd {dir}
docker compose exec -T api python - <<'PY'
import json, urllib.request
request = urllib.request.Request("http://127.0.0.1:8010/api/health", headers={headers})
with urllib.request.urlopen(request, timeout=10) as response:
    print(json.dumps({{"body": json.loads(response.read()), "allow_origin": response.headers.get("access-control-allow-origin")}}))
PY
"""


def wait_for_api(remote, timeout=1200):
    """Wait until the API container's health check passes: it downloads the weights and loads the
    models on its first start, which takes a few minutes."""
    deadline = time.time() + timeout
    last = None
    while True:
        status = remote.run(API_HEALTH.format(dir=REMOTE_DIR), capture=True).strip()
        if status == "healthy":
            return
        if status in ("missing", "exited", "dead", "unhealthy"):
            raise RuntimeError(f"the API container is {status}; see `docker compose logs api` on the VM")
        if time.time() > deadline:
            raise RuntimeError(f"the API is still {status} after {timeout // 60} minutes; "
                               "see `docker compose logs api` on the VM")
        if status != last:
            log(f"  API container: {status} (it downloads the weights and loads the models first)")
            last = status
        time.sleep(15)


def api_request(remote, origin=None):
    """GET /api/health from inside the API container, optionally with an Origin header."""
    headers = repr({"Origin": origin} if origin else {})
    return json.loads(remote.run(API_REQUEST.format(dir=REMOTE_DIR, headers=headers), capture=True))
