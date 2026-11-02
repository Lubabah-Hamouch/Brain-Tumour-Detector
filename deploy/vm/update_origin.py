"""Allow the web client's address to call the API: set ALLOWED_ORIGIN on the VM (DEPLOYMENT.md, section 9).

    python deploy/vm/update_origin.py https://brain-tumour-detector.vercel.app

Several addresses can be given, separated by spaces or commas; they replace the current list.
Over SSH, it sets ALLOWED_ORIGIN in ~/brain-tumour-api/.env and recreates only the API container
with it (docker compose up -d --no-build --no-deps api): no rebuild, and Caddy keeps running with
its certificate. A plain `docker compose restart` would not do: a container keeps the environment
it was created with. The new container downloads the weights again, so the API is back after
about a minute; then the script checks that it answers the first address with the CORS header.

The VM's IP comes from --ip or INSTANCE_IP in deploy/vm/instance.env.
"""

import argparse
import sys

from common import (REMOTE_DIR, Remote, api_request, check_ip, check_origins, fail, log, parse_env,
                    read_settings, ssh_private_key, wait_for_api)

RECREATE = f"""
set -euo pipefail
cd {REMOTE_DIR}
docker compose up -d --no-build --no-deps api
"""


def set_line(text, key, value):
    """Set KEY=value in .env text, keeping the other lines."""
    lines, found = [], False
    for line in text.splitlines():
        if line.startswith(f"{key}="):
            line, found = f"{key}={value}", True
        lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Set ALLOWED_ORIGIN on the VM and recreate the API container.")
    parser.add_argument("origins", nargs="+",
                        help="the web client's address(es), e.g. https://brain-tumour-detector.vercel.app")
    parser.add_argument("--ip", help="the VM's public IP (default: INSTANCE_IP in instance.env)")
    parser.add_argument("--user", default="ubuntu", help="SSH user (default: ubuntu)")
    args = parser.parse_args()

    origins = check_origins(",".join(args.origins))
    if not origins:
        fail("no origin given")
    settings = read_settings()
    ip = args.ip or settings.get("INSTANCE_IP")
    if not ip:
        fail("no IP: pass --ip, or set INSTANCE_IP in instance.env.")
    remote = Remote(check_ip(ip), ssh_private_key(settings), args.user)

    try:
        current = remote.run(f"cat {REMOTE_DIR}/.env", capture=True)
    except RuntimeError as error:
        fail(f"could not read {REMOTE_DIR}/.env on the VM ({error}). Has setup_remote.py run?")
    before = parse_env(current).get("ALLOWED_ORIGIN", "")
    log(f"ALLOWED_ORIGIN: {before or '(empty)'} -> {origins}")
    if before == origins:
        log("Already set; the API container is left as it is.")
    else:
        remote.write_file(f"{REMOTE_DIR}/.env", set_line(current, "ALLOWED_ORIGIN", origins))
        log("Recreating the API container with the new setting (Caddy keeps running)...")
        remote.run(RECREATE)

    try:
        wait_for_api(remote)
        first = origins.split(",")[0]
        answer = api_request(remote, origin=first)
    except RuntimeError as error:
        fail(str(error))
    if answer["allow_origin"] != first:
        fail(f"the API does not allow {first} (access-control-allow-origin: {answer['allow_origin']}). "
             "See `docker compose logs api` on the VM.")
    log(f"Done: the API answers {first} with access-control-allow-origin: {answer['allow_origin']}.")
    domain = parse_env(current).get("DOMAIN")
    if domain:
        print(f"\nCheck in the browser: open {first}, then DEPLOYMENT.md, section 10 (API: https://{domain}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
