"""Set up the VM over SSH and start the API on it: DEPLOYMENT.md, section 6, as one command.

Run it on your machine, from the repository root, in the project environment (it builds the
bundle with build_bundle.py, which needs it):

    python deploy/vm/setup_remote.py --domain brain-tumour-api.duckdns.org

In stages, each reported as it finishes, stopping at the first that fails:

    1. Connect     wait until the VM accepts SSH (a new instance takes a minute or two)
    2. System      wait for the first boot's setup, update the packages; add a 4 GB swap file if
                   the VM has less than 10 GB of memory (the 1 OCPU / 6 GB shape) or with --swap
    3. Firewall    open ports 80 and 443 in the VM's own iptables rules, and save them for reboots
    4. Docker      install Docker Engine and Docker Compose from Docker's repository
    5. Bundle      build the bundle here (build_bundle.py; skip with --skip-build)
    6. Upload      copy build/brain-tumour-api.tar.gz to the VM
    7. Start       unpack it into ~/brain-tumour-api, write .env (DOMAIN, ALLOWED_ORIGIN), and
                   run docker compose up -d --build
    8. Check       wait until the API is healthy, then check it from inside the VM, port 80 from
                   here, and HTTPS if the domain already points at the VM

Every stage is safe to run again, so after a failure, fix the cause and run the same command
(with --skip-build if the bundle is already built). It also serves to update the API later.

The VM's IP comes from --ip or INSTANCE_IP in deploy/vm/instance.env (create_instance.py writes
it); the domain from --domain or DOMAIN there (saved on the first run). ALLOWED_ORIGIN is --origin
if given, or else the value already on the VM, or else empty until update_origin.py sets it.
"""

import argparse
import http.client
import json
import socket
import ssl
import subprocess
import sys
import time
import urllib.request

from common import (ARCHIVE, HERE, REMOTE_DIR, ROOT, Remote, api_request, check_domain, check_ip,
                    check_origins, fail, log, parse_env, read_settings, ssh_private_key, wait_for_api,
                    write_settings)

APT = ("sudo env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get -y -q "
       "-o DPkg::Lock::Timeout=600 -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold")

SYSTEM = f"""
set -euo pipefail
echo "Waiting for the first boot's setup (cloud-init) to finish..."
cloud-init status --wait >/dev/null 2>&1 || true
echo "Updating the packages (a few minutes the first time)..."
{APT} update >/dev/null
{APT} upgrade >/dev/null
{APT} install ca-certificates curl >/dev/null
if [ -f /var/run/reboot-required ]; then echo "Note: the update asks for a reboot (sudo reboot); the API comes back by itself."; fi
"""

SWAP = """
set -euo pipefail
if [ -n "$(swapon --show --noheadings)" ]; then
  echo "Swap is already on:"; swapon --show
else
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  echo "Added a 4 GB swap file, also used after a reboot."
fi
"""

# Oracle's Ubuntu images reject everything but SSH in their own iptables rules, kept in
# /etc/iptables/rules.v4. Insert the rule above the REJECT, both live and in that file. Editing the
# file instead of `netfilter-persistent save` keeps Docker's own rules out of it.
FIREWALL = r"""
set -euo pipefail
RULE="-p tcp -m multiport --dports 80,443 -m conntrack --ctstate NEW -j ACCEPT"
if sudo iptables -C INPUT $RULE 2>/dev/null; then
  echo "Live rules: ports 80 and 443 already open"
else
  sudo iptables -I INPUT $RULE
  echo "Live rules: opened ports 80 and 443"
fi
FILE=/etc/iptables/rules.v4
if [ ! -f "$FILE" ]; then
  echo "No $FILE on this image: nothing to save, the rule above applies until a reboot."
elif sudo grep -qF -- "-A INPUT $RULE" "$FILE"; then
  echo "Saved rules: ports 80 and 443 already open"
elif sudo grep -q -- '^-A INPUT -j REJECT' "$FILE"; then
  sudo cp "$FILE" "$FILE.bak"
  sudo sed -i "0,/^-A INPUT -j REJECT/s//-A INPUT $RULE\n&/" "$FILE"
  if sudo iptables-restore --test < "$FILE"; then
    echo "Saved rules: opened ports 80 and 443 (kept after a reboot; previous file in $FILE.bak)"
  else
    sudo mv "$FILE.bak" "$FILE"; echo "The edited $FILE did not load; restored it." >&2; exit 1
  fi
else
  echo "Saved rules: no REJECT rule in $FILE, nothing to add"
fi
accept=$(sudo iptables -L INPUT -n --line-numbers | awk '/ACCEPT/ && /dports 80,443/ {print $1; exit}')
reject=$(sudo iptables -L INPUT -n --line-numbers | awk '/REJECT/ {print $1; exit}')
if [ -n "$reject" ] && [ "$accept" -gt "$reject" ]; then
  echo "The ACCEPT rule for 80,443 (line $accept) is below the REJECT rule (line $reject)." >&2; exit 1
fi
"""

DOCKER = f"""
set -euo pipefail
if command -v docker >/dev/null && docker compose version >/dev/null 2>&1; then
  echo "Docker is already installed."
else
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${{UBUNTU_CODENAME:-$VERSION_CODENAME}}") stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  {APT} update >/dev/null
  {APT} install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin >/dev/null
fi
sudo systemctl enable --now docker >/dev/null 2>&1
# Lets the ubuntu user run docker without sudo, from its next login (each stage is a new one).
sudo usermod -aG docker ubuntu
sudo docker --version
sudo docker compose version
"""

UNPACK = f"""
set -euo pipefail
rm -rf {REMOTE_DIR}.new && mkdir {REMOTE_DIR}.new
tar -xzf ~/brain-tumour-api.tar.gz -C {REMOTE_DIR}.new --strip-components=1
rm -rf {REMOTE_DIR} && mv {REMOTE_DIR}.new {REMOTE_DIR}
echo "Unpacked into {REMOTE_DIR}"
"""

START = f"""
set -euo pipefail
cd {REMOTE_DIR}
docker compose up -d --build
docker image prune -f >/dev/null
docker compose ps
"""


def env_file(domain, origin):
    """The VM's .env: deploy/vm/.env.example with the two values set."""
    lines = []
    for line in (HERE / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.startswith("DOMAIN="):
            line = f"DOMAIN={domain}"
        elif line.startswith("ALLOWED_ORIGIN="):
            line = f"ALLOWED_ORIGIN={origin}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def resolves_to(domain, ip):
    try:
        return ip in socket.gethostbyname_ex(domain)[2]
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Set up the VM over SSH and start the API on it.",
                                     epilog="DEPLOYMENT.md, section 6, describes each stage.")
    parser.add_argument("--ip", help="the VM's public IP (default: INSTANCE_IP in instance.env)")
    parser.add_argument("--domain", help="the API's host name, e.g. brain-tumour-api.duckdns.org "
                                         "(default: DOMAIN in instance.env)")
    parser.add_argument("--origin", help="ALLOWED_ORIGIN: the web client's https:// address, if you "
                                         "already have it (default: keep the VM's, or leave it empty)")
    parser.add_argument("--skip-build", action="store_true",
                        help="use the existing build/brain-tumour-api.tar.gz instead of building it")
    parser.add_argument("--swap", action="store_true", help="add a 4 GB swap file whatever the memory")
    parser.add_argument("--user", default="ubuntu", help="SSH user (default: ubuntu)")
    args = parser.parse_args()

    settings = read_settings()
    ip = args.ip or settings.get("INSTANCE_IP")
    if not ip:
        fail("no IP: pass --ip, or run create_instance.py, which saves it in instance.env.")
    ip = check_ip(ip)
    domain = args.domain or settings.get("DOMAIN")
    if not domain and sys.stdin.isatty():
        domain = input("The API's DuckDNS name (e.g. brain-tumour-api.duckdns.org): ")
    if not domain:
        fail("no domain: pass --domain, e.g. --domain brain-tumour-api.duckdns.org (DEPLOYMENT.md, section 5).")
    domain = check_domain(domain)
    origin = check_origins(args.origin) if args.origin else None
    write_settings({"INSTANCE_IP": ip, "DOMAIN": domain})

    remote = Remote(ip, ssh_private_key(settings), args.user)
    log(f"Setting up {args.user}@{ip} for {domain}")
    if not resolves_to(domain, ip):
        log(f"Note: {domain} does not resolve to {ip} yet (DEPLOYMENT.md, section 5). Setup goes on; "
            "HTTPS starts working once it does.")

    def connect():
        remote.wait_until_reachable()
        return remote.run("uname -m; . /etc/os-release; echo $PRETTY_NAME", capture=True).replace("\n", ", ").strip(", ")

    def system():
        remote.run(SYSTEM)
        memory_kb = int(remote.run("awk '/MemTotal/ {print $2}' /proc/meminfo", capture=True))
        memory = memory_kb / 2**20
        if args.swap or memory < 10:
            log(f"  {memory:.1f} GiB of memory: adding swap (DEPLOYMENT.md, Sizing)")
            remote.run(SWAP)
        return f"{memory:.1f} GiB of memory"

    def firewall():
        remote.run(FIREWALL)

    def docker():
        remote.run(DOCKER)

    def bundle():
        if args.skip_build:
            if not ARCHIVE.exists():
                raise RuntimeError(f"--skip-build, but {ARCHIVE.relative_to(ROOT)} does not exist")
            return f"using the existing {ARCHIVE.relative_to(ROOT)}"
        subprocess.run([sys.executable, str(HERE / "build_bundle.py")], cwd=ROOT, check=True)
        return str(ARCHIVE.relative_to(ROOT))

    def upload():
        remote.upload(ARCHIVE, "brain-tumour-api.tar.gz")
        return f"{ARCHIVE.stat().st_size / 2**20:.0f} MB"

    def start():
        nonlocal origin
        if origin is None:
            current = remote.run(f"cat {REMOTE_DIR}/.env 2>/dev/null || true", capture=True)
            origin = parse_env(current).get("ALLOWED_ORIGIN", "")
        remote.run(UNPACK)
        remote.write_file(f"{REMOTE_DIR}/.env", env_file(domain, origin))
        log(f"  .env: DOMAIN={domain}, ALLOWED_ORIGIN={origin or '(empty: update_origin.py sets it)'}")
        log("  docker compose up -d --build: the first build takes about 5-10 minutes")
        remote.run(START)

    def check():
        wait_for_api(remote)
        health = api_request(remote)["body"]
        log(f"  inside the VM: /api/health -> {json.dumps(health)}")
        if health.get("status") != "ok":
            raise RuntimeError("the API answers, but its status is not ok")
        notes = [f"API healthy, sample results {health.get('sample_results')}"]

        # Port 80 from here: Caddy answers with its redirect to HTTPS, which shows that Oracle's
        # security list and the VM's firewall let the traffic through.
        try:
            connection = http.client.HTTPConnection(ip, 80, timeout=15)
            connection.request("GET", "/api/health", headers={"Host": domain})
            response = connection.getresponse()
            log(f"  from here: http://{ip}/api/health (Host: {domain}) -> {response.status} "
                f"{response.getheader('Location') or ''}")
            notes.append("port 80 reachable")
        except OSError as error:
            raise RuntimeError(f"port 80 is not reachable from here ({error}): check the security list "
                               "(create_network.py) and the firewall stage") from error

        if not resolves_to(domain, ip):
            notes.append(f"HTTPS waits for DNS: point {domain} at {ip} (section 5); Caddy then gets "
                         "its certificate within a minute")
            return "; ".join(notes)
        url = f"https://{domain}/api/health"
        for attempt in range(12):
            try:
                with urllib.request.urlopen(url, timeout=15) as response:
                    log(f"  from here: {url} -> {response.status}")
                notes.append("HTTPS works")
                return "; ".join(notes)
            except (OSError, ssl.SSLError) as error:
                if attempt == 0:
                    log(f"  waiting for Caddy's certificate ({error})")
                time.sleep(15)
        notes.append("HTTPS not working yet after 3 minutes: see `docker compose logs caddy` on the VM")
        return "; ".join(notes)

    stages = [("Connect", connect), ("System", system), ("Firewall", firewall), ("Docker", docker),
              ("Bundle", bundle), ("Upload", upload), ("Start", start), ("Check", check)]
    results = []
    try:
        for number, (name, stage) in enumerate(stages, 1):
            print(f"\n=== [{number}/{len(stages)}] {name}", flush=True)
            try:
                results.append((name, "ok", stage() or ""))
            except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
                results.append((name, "FAILED", str(error)))
                break
    except KeyboardInterrupt:
        results.append((stages[len(results)][0], "stopped", "Ctrl+C"))

    print("\n=== Summary")
    for name, _ in stages:
        status, detail = next(((s, d) for n, s, d in results if n == name), ("not run", ""))
        print(f"  {name:<9} {status:<8} {detail}")
    if all(status == "ok" for _, status, _ in results) and len(results) == len(stages):
        print(f"\nThe API runs on the VM. Next: the Vercel project (DEPLOYMENT.md, section 8), "
              f"with NEXT_PUBLIC_API_URL=https://{domain}")
        return 0
    print("\nFix the cause and run the same command again (add --skip-build if the bundle stage passed).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
