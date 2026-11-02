"""Create the Always Free Ampere VM with the OCI CLI, retrying while the region is out of capacity.

Creating a VM.Standard.A1.Flex instance often fails with "Out of host capacity" in busy regions.
Capacity comes and goes, so this script tries the launch again every 60-90 seconds until it
succeeds, prints the instance's details and public IP when it does, and saves the IP in
deploy/vm/instance.env as INSTANCE_IP for setup_remote.py. DEPLOYMENT.md (sections 2-4) explains
how to set up the OCI CLI; create_network.py fills in the settings this script needs.

It only ever asks for an Always Free shape: 2 OCPUs and 12 GB, or, with your agreement,
1 OCPU and 6 GB (--fallback-after, --small). It stops before launching if the compartment already
has an A1 instance, so it cannot create a second one beside it, and it stops on any error other
than capacity, rate limiting or a temporary server error, instead of retrying it.

The settings come from deploy/vm/instance.env (written by create_network.py, or copied from
instance.env.example; it is gitignored) or from environment variables of the same names, which
take precedence:

    OCI_COMPARTMENT_ID        where the instance goes (the tenancy OCID for the root compartment)
    OCI_SUBNET_ID             the public subnet of your VCN
    SSH_PUBLIC_KEY_FILE       e.g. ~/.ssh/oracle_vm.pub
    OCI_IMAGE_ID              optional: default is the newest Canonical Ubuntu 24.04 aarch64 image
    OCI_AVAILABILITY_DOMAINS  optional, comma-separated: default is all of the region's, in turn
    INSTANCE_NAME             optional: default brain-tumour-api

The OCI CLI's own settings apply as usual (~/.oci/config; OCI_CLI_PROFILE picks a profile).
From the repository root:

    python deploy/vm/create_instance.py --dry-run     # check the settings, launch nothing
    python deploy/vm/create_instance.py               # retry until the instance is created

Stop it at any time with Ctrl+C.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from common import SETTINGS_FILE, OciError, fail, log, oci, read_settings, write_settings

SHAPE = "VM.Standard.A1.Flex"
FULL = {"ocpus": 2, "memoryInGBs": 12}   # the whole Always Free Ampere allowance
SMALL = {"ocpus": 1, "memoryInGBs": 6}
MIN_INTERVAL = 60                        # never retry faster than this, in seconds
MAX_BACKOFF = 300                        # longest wait after rate limiting


def need(settings, key):
    if key not in settings:
        fail(f"{key} is not set. Run create_network.py, which writes it to {SETTINGS_FILE.name}, "
             "or set it there (see instance.env.example) or as an environment variable.")
    return settings[key]


def save_ip(path, ip):
    """Remember the IP for setup_remote.py, in the settings file if one is in use."""
    if ip and path.exists():
        write_settings({"INSTANCE_IP": ip}, path)
        log(f"Saved INSTANCE_IP={ip} in {path.name}.")


def ssh_key_file(settings):
    path = Path(os.path.expanduser(need(settings, "SSH_PUBLIC_KEY_FILE")))
    if not path.is_file():
        fail(f"SSH public key not found: {path}")
    if not path.read_text(encoding="utf-8").startswith(("ssh-", "ecdsa-")):
        fail(f"{path} is not an OpenSSH public key. Use the .pub file, never the private key.")
    return path


def find_image(compartment):
    images = oci("compute", "image", "list", "--compartment-id", compartment,
                 "--operating-system", "Canonical Ubuntu", "--operating-system-version", "24.04",
                 "--shape", SHAPE, "--lifecycle-state", "AVAILABLE",
                 "--sort-by", "TIMECREATED", "--sort-order", "DESC", "--all")
    for image in (images or {}).get("data", []):
        name = image["display-name"]
        if "aarch64" in name and "Minimal" not in name:
            return image["id"], name
    fail("no Canonical Ubuntu 24.04 aarch64 image found; set OCI_IMAGE_ID.")


def existing_a1_instances(compartment):
    instances = oci("compute", "instance", "list", "--compartment-id", compartment, "--all")
    return [i for i in (instances or {}).get("data", [])
            if i["shape"] == SHAPE and i["lifecycle-state"] not in ("TERMINATED", "TERMINATING")]


def describe(instance):
    config = instance.get("shape-config") or {}
    return (f"{instance['display-name']} ({instance['lifecycle-state']}, "
            f"{config.get('ocpus', '?')} OCPU / {config.get('memory-in-gbs', '?')} GB, "
            f"{instance['availability-domain']})\n  {instance['id']}")


def wait_until_running(instance_id):
    """Poll the new instance until it runs. A launch that is accepted can still end TERMINATED.

    Errors while polling are retried here, never by launching again, so that a temporary error
    cannot lead to a second instance. Any other error ends the script with the instance's OCID.
    """
    while True:
        try:
            instance = oci("compute", "instance", "get", "--instance-id", instance_id)["data"]
        except OciError as error:
            if error.kind == "fatal":
                fail(f"could not check the new instance {instance_id}:\n{error}\n"
                     "Look it up in the console (Compute > Instances).")
            log(f"  could not check the instance ({error}); checking again")
            time.sleep(30)
            continue
        state = instance["lifecycle-state"]
        if state == "RUNNING":
            return instance
        if state in ("TERMINATING", "TERMINATED"):
            return None
        log(f"  state: {state}, waiting...")
        time.sleep(15)


def public_ip(instance):
    vnics = oci("compute", "instance", "list-vnics", "--instance-id", instance["id"],
                "--compartment-id", instance["compartment-id"])
    for vnic in (vnics or {}).get("data", []):
        if vnic.get("public-ip"):
            return vnic["public-ip"]
    return None


def ask(question):
    if not sys.stdin.isatty():
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create the Always Free Ampere VM, retrying while the region is out of capacity.")
    parser.add_argument("--config", type=Path, default=SETTINGS_FILE,
                        help=f"settings file (default: deploy/vm/{SETTINGS_FILE.name})")
    parser.add_argument("--interval", type=int, default=75,
                        help=f"average seconds between attempts, at least {MIN_INTERVAL} (default 75; "
                             "each wait varies by up to 15 s either way)")
    parser.add_argument("--fallback-after", type=int, default=60, metavar="N",
                        help="after N failed attempts at 2 OCPU / 12 GB, offer to switch to 1 OCPU / 6 GB "
                             "(default 60, about 75 minutes; 0 never offers)")
    parser.add_argument("--auto-fallback", action="store_true",
                        help="switch to 1 OCPU / 6 GB after --fallback-after attempts without asking")
    parser.add_argument("--small", action="store_true", help="ask for 1 OCPU / 6 GB from the start")
    parser.add_argument("--max-attempts", type=int, default=0, metavar="N",
                        help="give up after N attempts (default 0: keep trying until stopped)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the settings and print the launch command, but launch nothing")
    args = parser.parse_args()
    if args.interval < MIN_INTERVAL:
        parser.error(f"--interval must be at least {MIN_INTERVAL} seconds")

    settings = read_settings(args.config)
    compartment = need(settings, "OCI_COMPARTMENT_ID")
    subnet = need(settings, "OCI_SUBNET_ID")
    key_file = ssh_key_file(settings)
    name = settings.get("INSTANCE_NAME") or "brain-tumour-api"

    try:
        # Never start a second A1 instance: 2 OCPU / 12 GB is the whole Always Free allowance.
        existing = existing_a1_instances(compartment)
        if existing:
            print("This compartment already has an Ampere A1 instance, so none is created:")
            for instance in existing:
                ip = public_ip(instance)
                print("  " + describe(instance))
                print(f"  Public IP: {ip or 'none'}")
                if instance["display-name"] == name and not args.dry_run:
                    save_ip(args.config, ip)
            print("Use it (DEPLOYMENT.md, section 5 onwards), or terminate it first if it is not wanted.")
            return 0

        if settings.get("OCI_IMAGE_ID"):
            image, image_name = settings["OCI_IMAGE_ID"], "OCI_IMAGE_ID"
        else:
            image, image_name = find_image(compartment)
        domains = [d.strip() for d in settings.get("OCI_AVAILABILITY_DOMAINS", "").split(",") if d.strip()]
        if not domains:
            listed = oci("iam", "availability-domain", "list", "--compartment-id", compartment)
            domains = [d["name"] for d in listed["data"]]
    except OciError as error:
        fail(f"the OCI CLI could not read your account:\n{error}\n"
             "Check ~/.oci/config with `oci iam region list` (DEPLOYMENT.md, section 2).")

    shape_config = SMALL if args.small else FULL
    log(f"Instance {name}: {SHAPE}, image {image_name}")
    log(f"Availability domains, tried in turn: {', '.join(domains)}")

    def launch_args(domain):
        return ["compute", "instance", "launch",
                "--display-name", name,
                "--compartment-id", compartment,
                "--availability-domain", domain,
                "--shape", SHAPE,
                "--shape-config", json.dumps(shape_config),
                "--image-id", image,
                "--subnet-id", subnet,
                "--assign-public-ip", "true",
                "--ssh-authorized-keys-file", str(key_file)]

    if args.dry_run:
        print("\nDry run. The launch command would be:\n  oci " + " ".join(launch_args(domains[0])))
        return 0

    log(f"Trying {shape_config['ocpus']} OCPU / {shape_config['memoryInGBs']} GB every "
        f"~{args.interval} s. Press Ctrl+C to stop.")
    attempt, full_failures, wait = 0, 0, args.interval
    try:
        while True:
            attempt += 1
            domain = domains[(attempt - 1) % len(domains)]
            size = f"{shape_config['ocpus']} OCPU / {shape_config['memoryInGBs']} GB"
            log(f"Attempt {attempt}: {size} in {domain}")
            try:
                launched = oci(*launch_args(domain))["data"]
                log(f"  accepted: {launched['id']}")
                instance = wait_until_running(launched["id"])
                if instance is None:
                    log("  the instance was terminated while provisioning (capacity); trying again")
                    raise OciError("capacity", "terminated while provisioning")
                break
            except OciError as error:
                if error.kind == "fatal":
                    log("  failed with an error that retrying will not fix:")
                    print(str(error), file=sys.stderr)
                    return 1
                if error.kind == "rate_limit":
                    wait = min(wait * 2, MAX_BACKOFF)
                    log(f"  rate limited by the API; slowing down to ~{wait} s")
                elif error.kind == "transient":
                    log(f"  temporary error ({error}); trying again")
                else:
                    wait = args.interval
                    log("  out of host capacity")

            if shape_config is FULL:
                full_failures += 1
                if args.fallback_after and full_failures == args.fallback_after:
                    question = (f"{full_failures} attempts at 2 OCPU / 12 GB failed. Switch to "
                                "1 OCPU / 6 GB? It is enough with the swap file from step 5.4, "
                                "but segmentations take twice as long.")
                    if args.auto_fallback or ask(question):
                        shape_config = SMALL
                        log("Switching to 1 OCPU / 6 GB.")
                    else:
                        log("Staying with 2 OCPU / 12 GB.")
            if args.max_attempts and attempt >= args.max_attempts:
                log(f"Giving up after {attempt} attempts. Nothing was created.")
                return 1
            pause = max(MIN_INTERVAL, wait + random.uniform(-15, 15))
            log(f"  next attempt in {pause:.0f} s")
            time.sleep(pause)
    except KeyboardInterrupt:
        print()
        log(f"Stopped after {attempt} attempt(s).")
        log("If you stopped it while an attempt was running, check Compute > Instances in the "
            "console; running this script again also reports an instance that was created.")
        return 130

    ip = public_ip(instance)
    print()
    log("Instance created and running:")
    print("  " + describe(instance))
    print(f"  Public IP: {ip or 'none (check that the subnet is public)'}")
    if ip:
        print(f"\nConnect with:  ssh -i {key_file.with_suffix('')} ubuntu@{ip}")
        save_ip(args.config, ip)
    print("Continue with DEPLOYMENT.md, section 5 (point the DuckDNS name at this IP).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
