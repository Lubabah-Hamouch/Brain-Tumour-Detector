"""Create the network for the VM with the OCI CLI, and the settings create_instance.py needs.

It creates what the console's "Create VCN with Internet Connectivity" wizard creates for the
public side, with the ports the API needs already open:

    brain-tumour-vcn            a VCN, 10.0.0.0/16
    brain-tumour-igw            an internet gateway
    brain-tumour-public-rt      a route table sending 0.0.0.0/0 to the internet gateway
    brain-tumour-public-sl      a security list: TCP 22 (SSH), 80 and 443 (Caddy) from anywhere,
                                ICMP for path MTU discovery, and all outbound traffic
    brain-tumour-public-subnet  a public subnet, 10.0.0.0/24, using the route table and list

It is safe to run again: it looks each resource up by name first, reuses it if it exists, and adds
only rules that are missing. It never deletes anything. All of these are free.

Then it writes the settings to deploy/vm/instance.env (created from instance.env.example if
needed): OCI_COMPARTMENT_ID (the tenancy, i.e. the root compartment, from ~/.oci/config unless
set), OCI_SUBNET_ID and SSH_PUBLIC_KEY_FILE. If that SSH key (default ~/.ssh/oracle_vm.pub) does
not exist, it creates the pair with ssh-keygen.

From the repository root (DEPLOYMENT.md, section 3):

    python deploy/vm/create_network.py --dry-run    # look up only, change nothing
    python deploy/vm/create_network.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import (DEFAULT_SSH_KEY, SETTINGS_FILE, OciError, fail, log, oci, oci_config_tenancy,
                    read_settings, write_settings)

VCN = "brain-tumour-vcn"
GATEWAY = "brain-tumour-igw"
ROUTE_TABLE = "brain-tumour-public-rt"
SECURITY_LIST = "brain-tumour-public-sl"
SUBNET = "brain-tumour-public-subnet"
VCN_CIDR = "10.0.0.0/16"
SUBNET_CIDR = "10.0.0.0/24"
ANYWHERE = "0.0.0.0/0"


def tcp_in(port, description):
    return {"protocol": "6", "source": ANYWHERE, "sourceType": "CIDR_BLOCK", "isStateless": False,
            "tcpOptions": {"destinationPortRange": {"min": port, "max": port}}, "description": description}


def icmp_in(source, icmp_type, code=None):
    options = {"type": icmp_type, **({"code": code} if code is not None else {})}
    return {"protocol": "1", "source": source, "sourceType": "CIDR_BLOCK", "isStateless": False,
            "icmpOptions": options}


INGRESS = [
    tcp_in(22, "SSH"),
    tcp_in(80, "Caddy: HTTP, for Let's Encrypt and the redirect to HTTPS"),
    tcp_in(443, "Caddy: HTTPS"),
    # As in Oracle's default security list: path MTU discovery, and unreachable messages in the VCN.
    icmp_in(ANYWHERE, 3, 4),
    icmp_in(VCN_CIDR, 3),
]
EGRESS = [{"protocol": "all", "destination": ANYWHERE, "destinationType": "CIDR_BLOCK", "isStateless": False}]


def camel(value):
    """The CLI prints kebab-case keys but takes camelCase in JSON arguments; nulls are dropped."""
    if isinstance(value, dict):
        return {k.split("-")[0] + "".join(p.title() for p in k.split("-")[1:]): camel(v)
                for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [camel(v) for v in value]
    return value


def rule_key(rule):
    """What makes two security rules the same, for rules in camelCase."""
    ports = (rule.get("tcpOptions") or {}).get("destinationPortRange") or {}
    icmp = rule.get("icmpOptions") or {}
    return (rule["protocol"], rule.get("source") or rule.get("destination"),
            ports.get("min"), ports.get("max"), icmp.get("type"), icmp.get("code"))


class Network:
    def __init__(self, compartment, dry_run):
        self.compartment = compartment
        self.dry_run = dry_run

    def find(self, resource, name, vcn_id=None):
        """The one live resource of this kind and name, or None."""
        args = ["network", resource, "list", "--compartment-id", self.compartment,
                "--display-name", name, "--all"]
        if vcn_id:
            args += ["--vcn-id", vcn_id]
        found = [r for r in ((oci(*args) or {}).get("data") or [])
                 if r["lifecycle-state"] not in ("TERMINATING", "TERMINATED")]
        if len(found) > 1:
            fail(f"there are {len(found)} {resource}s named {name}; delete the extra ones in the console "
                 "(Networking), then run this again.")
        if found and found[0]["lifecycle-state"] != "AVAILABLE":
            fail(f"{resource} {name} is {found[0]['lifecycle-state']}; wait a minute and run this again.")
        return found[0] if found else None

    def create(self, label, name, *args):
        if self.dry_run:
            log(f"{label} {name}: would be created")
            return None
        created = oci("network", *args, "--compartment-id", self.compartment, "--display-name", name,
                      "--wait-for-state", "AVAILABLE")["data"]
        log(f"{label} {name}: created ({created['id']})")
        return created

    def change(self, label, name, what, *args):
        if self.dry_run:
            log(f"{label} {name}: exists; would {what}")
            return
        oci("network", *args, "--force", "--wait-for-state", "AVAILABLE")
        log(f"{label} {name}: {what} - done")

    def vcn(self):
        vcn = self.find("vcn", VCN)
        if vcn:
            log(f"VCN {VCN}: exists ({vcn['id']})")
            return vcn
        return self.create("VCN", VCN, "vcn", "create", "--cidr-blocks", json.dumps([VCN_CIDR]),
                           "--dns-label", "braintumour")

    def gateway(self, vcn):
        gateway = self.find("internet-gateway", GATEWAY, vcn["id"])
        if gateway:
            if not gateway.get("is-enabled"):
                fail(f"internet gateway {GATEWAY} is disabled; enable it in the console (Networking, "
                     f"Virtual cloud networks, {VCN}, Internet gateways), then run this again.")
            log(f"Internet gateway {GATEWAY}: exists ({gateway['id']})")
            return gateway
        return self.create("Internet gateway", GATEWAY, "internet-gateway", "create",
                           "--vcn-id", vcn["id"], "--is-enabled", "true")

    def route_table(self, vcn, gateway):
        default_route = {"destination": ANYWHERE, "destinationType": "CIDR_BLOCK",
                         "networkEntityId": gateway["id"] if gateway else None,
                         "description": "All other traffic to the internet gateway"}
        table = self.find("route-table", ROUTE_TABLE, vcn["id"])
        if not table:
            if gateway is None:  # dry run, gateway still to be created
                return self.create("Route table", ROUTE_TABLE)
            return self.create("Route table", ROUTE_TABLE, "route-table", "create", "--vcn-id", vcn["id"],
                               "--route-rules", json.dumps([default_route]))
        rules = [{k: v for k, v in camel(r).items() if k in ("destination", "destinationType",
                                                              "networkEntityId", "description")}
                 for r in table.get("route-rules") or []]
        if gateway is None:  # dry run, gateway still to be created
            self.change("Route table", ROUTE_TABLE, f"route {ANYWHERE} to the new internet gateway")
        elif any(r.get("destination") == ANYWHERE and r.get("networkEntityId") == gateway["id"] for r in rules):
            log(f"Route table {ROUTE_TABLE}: exists, routes {ANYWHERE} to the internet gateway")
        else:
            # Replace any other default route: the subnet must reach the internet through the gateway.
            rules = [r for r in rules if r.get("destination") != ANYWHERE] + [default_route]
            self.change("Route table", ROUTE_TABLE, f"route {ANYWHERE} to the internet gateway",
                        "route-table", "update", "--rt-id", table["id"], "--route-rules", json.dumps(rules))
        return table

    def security_list(self, vcn):
        security_list = self.find("security-list", SECURITY_LIST, vcn["id"])
        if not security_list:
            return self.create("Security list", SECURITY_LIST, "security-list", "create",
                               "--vcn-id", vcn["id"], "--ingress-security-rules", json.dumps(INGRESS),
                               "--egress-security-rules", json.dumps(EGRESS))
        ingress = camel(security_list.get("ingress-security-rules") or [])
        egress = camel(security_list.get("egress-security-rules") or [])
        have = {rule_key(r) for r in ingress} | {rule_key(r) for r in egress}
        missing_in = [r for r in INGRESS if rule_key(r) not in have]
        missing_out = [r for r in EGRESS if rule_key(r) not in have]
        if not missing_in and not missing_out:
            log(f"Security list {SECURITY_LIST}: exists, with all the rules")
        else:
            what = ", ".join(r.get("description") or f"protocol {r['protocol']}" for r in missing_in + missing_out)
            self.change("Security list", SECURITY_LIST, f"add the missing rules ({what})",
                        "security-list", "update", "--security-list-id", security_list["id"],
                        "--ingress-security-rules", json.dumps(ingress + missing_in),
                        "--egress-security-rules", json.dumps(egress + missing_out))
        return security_list

    def subnet(self, vcn, table, security_list):
        subnet = self.find("subnet", SUBNET, vcn["id"])
        if not subnet:
            return self.create("Subnet", SUBNET, "subnet", "create", "--vcn-id", vcn["id"],
                               "--cidr-block", SUBNET_CIDR, "--dns-label", "public",
                               "--prohibit-public-ip-on-vnic", "false",
                               "--route-table-id", table["id"] if table else None,
                               "--security-list-ids", json.dumps([security_list["id"] if security_list else None]))
        if subnet.get("prohibit-public-ip-on-vnic"):
            fail(f"subnet {SUBNET} exists but is private, which cannot be changed. Delete it in the "
                 "console, then run this again.")
        if table is None or security_list is None:  # dry run, still to be created
            self.change("Subnet", SUBNET, "use the new route table and security list")
        elif subnet.get("route-table-id") == table["id"] and security_list["id"] in subnet.get("security-list-ids", []):
            log(f"Subnet {SUBNET}: exists, public, with the route table and security list ({subnet['id']})")
        else:
            lists = list(dict.fromkeys(subnet.get("security-list-ids", []) + [security_list["id"]]))
            self.change("Subnet", SUBNET, "use the route table and security list",
                        "subnet", "update", "--subnet-id", subnet["id"], "--route-table-id", table["id"],
                        "--security-list-ids", json.dumps(lists))
        return subnet


def ssh_key(settings, dry_run):
    """Make sure the SSH key pair exists; return the public key's path as written in the settings."""
    setting = settings.get("SSH_PUBLIC_KEY_FILE", DEFAULT_SSH_KEY)
    public = Path(os.path.expanduser(setting))
    if public.suffix != ".pub":
        fail(f"SSH_PUBLIC_KEY_FILE must name the public key, ending in .pub: {setting}")
    private = public.with_suffix("")
    if public.exists():
        if not public.read_text(encoding="utf-8").startswith(("ssh-", "ecdsa-")):
            fail(f"{public} is not an OpenSSH public key.")
        log(f"SSH key {public}: exists")
    elif dry_run:
        log(f"SSH key {public}: would be {'derived from ' + str(private) if private.exists() else 'created'}")
    elif private.exists():
        # The private key is there without its .pub: write the public half from it.
        result = subprocess.run(["ssh-keygen", "-y", "-f", str(private)], capture_output=True, text=True)
        if result.returncode != 0:
            fail(f"could not read {private}: {result.stderr.strip()}")
        public.write_text(result.stdout, encoding="utf-8")
        log(f"SSH key {public}: written from the existing private key {private}")
    else:
        private.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "brain-tumour-api",
                        "-f", str(private)], check=True)
        log(f"SSH key {public}: created, with {private} (the private key: keep it, never share it)")
    return setting


def main():
    parser = argparse.ArgumentParser(description="Create the VM's network and write deploy/vm/instance.env.")
    parser.add_argument("--dry-run", action="store_true",
                        help="look up what exists and say what would be done, but change nothing")
    args = parser.parse_args()

    settings = read_settings()
    compartment = settings.get("OCI_COMPARTMENT_ID") or oci_config_tenancy()
    if not compartment:
        fail("no compartment: set OCI_COMPARTMENT_ID, or set up the OCI CLI first (DEPLOYMENT.md, section 2).")
    log(f"Compartment: {compartment}" + ("" if settings.get("OCI_COMPARTMENT_ID") else " (the tenancy, from ~/.oci/config)"))
    if args.dry_run:
        log("Dry run: nothing is created or changed.")

    network = Network(compartment, args.dry_run)
    try:
        vcn = network.vcn()
        if vcn is None:
            for label, name in [("Internet gateway", GATEWAY), ("Route table", ROUTE_TABLE),
                                ("Security list", SECURITY_LIST), ("Subnet", SUBNET)]:
                log(f"{label} {name}: would be created")
            subnet = None
        else:
            # In a dry run, a resource still to be created is None; the steps after it say so.
            gateway = network.gateway(vcn)
            table = network.route_table(vcn, gateway)
            security_list = network.security_list(vcn)
            subnet = network.subnet(vcn, table, security_list)
    except OciError as error:
        fail(f"the OCI CLI call failed:\n{error}\n"
             "Nothing is left half-made that a new run would duplicate: run this again after fixing the cause.")

    key_setting = ssh_key(settings, args.dry_run)
    updates = {"OCI_COMPARTMENT_ID": compartment, "SSH_PUBLIC_KEY_FILE": key_setting}
    if subnet:
        updates["OCI_SUBNET_ID"] = subnet["id"]
    if args.dry_run:
        log(f"Would write to {SETTINGS_FILE.name}: " + ", ".join(updates))
        return 0
    write_settings(updates)
    log(f"Wrote {', '.join(updates)} to deploy/vm/{SETTINGS_FILE.name}.")
    print("\nNetwork ready. Next: python deploy/vm/create_instance.py (DEPLOYMENT.md, section 4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
