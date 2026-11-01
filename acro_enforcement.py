
"""AURORA: ACRO network enforcement for Linux/Mininet-WiFi."""

from dataclasses import dataclass
import ipaddress
import subprocess

from aurora.acro.policy import Action, PolicyDecision


@dataclass(frozen=True)
class NetworkTarget:
    uav_id: int
    interface: str
    ip_address: str
    gateway: str = ""

    def __post_init__(self):
        if not self.interface or any(
            c not in "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
            for c in self.interface
        ):
            raise ValueError("Invalid interface name")
        ipaddress.ip_address(self.ip_address)
        if self.gateway:
            ipaddress.ip_address(self.gateway)


class NetworkEnforcer:
    """Apply ACRO decisions through Linux network commands."""

    def __init__(self, dry_run=True, rate_kbit=1000):
        if rate_kbit <= 0:
            raise ValueError("rate_kbit must be positive")
        self.dry_run = dry_run
        self.rate_kbit = rate_kbit
        self.targets = {}
        self.history = []

    def register(self, target: NetworkTarget):
        self.targets[target.uav_id] = target

    def _execute(self, command):
        if not self.dry_run:
            subprocess.run(command, check=True, timeout=10)
        self.history.append(command)
        return command

    def enforce(self, decision: PolicyDecision):
        """Enforce the action selected by the ACRO policy."""
        if decision.uav_id not in self.targets:
            raise KeyError("UAV network target not registered")

        target = self.targets[decision.uav_id]
        dev = target.interface
        ip = target.ip_address
        action = decision.action

        if action == Action.MONITOR:
            return []

        if action == Action.RESOURCE_ADJUST:
            # Resource scheduling requires a separate controller.
            return []

        if action == Action.RATE_LIMIT:
            commands = [[
                "tc", "qdisc", "replace", "dev", dev,
                "root", "tbf",
                "rate", f"{self.rate_kbit}kbit",
                "burst", "32kbit",
                "latency", "100ms",
            ]]

        elif action == Action.FILTER:
            commands = [[
                "iptables", "-I", "FORWARD",
                "-s", ip, "-j", "DROP",
            ]]

        elif action == Action.ISOLATE:
            commands = [[
                "ip", "link", "set", "dev", dev, "down",
            ]]

        elif action == Action.REROUTE:
            if not target.gateway:
                raise ValueError("Rerouting requires a gateway")
            commands = [[
                "ip", "route", "replace", "default",
                "via", target.gateway, "dev", dev,
            ]]

        else:
            raise ValueError(f"Unsupported action: {action}")

        return [self._execute(cmd) for cmd in commands]

    def restore(self, uav_id):
        """Restore interface connectivity and remove basic controls."""
        target = self.targets[uav_id]
        dev = target.interface
        ip = target.ip_address

        commands = [
            ["ip", "link", "set", "dev", dev, "up"],
            ["tc", "qdisc", "del", "dev", dev, "root"],
            ["iptables", "-D", "FORWARD", "-s", ip, "-j", "DROP"],
        ]

        results = []
        for cmd in commands:
            try:
                results.append(self._execute(cmd))
            except subprocess.CalledProcessError:
                pass  # Rule or qdisc may not exist.

        return results


if __name__ == "__main__":
    from aurora.acro.policy import ACROPolicyEngine

    policy = ACROPolicyEngine()
    enforcer = NetworkEnforcer(dry_run=True)

    enforcer.register(NetworkTarget(
        uav_id=1,
        interface="uav1-wlan0",
        ip_address="10.0.0.1",
        gateway="10.0.0.254",
    ))

    for trust in [0.9, 0.5, 0.2, 0.2, 0.2]:
        decision = policy.evaluate(1, trust)
        commands = enforcer.enforce(decision)
        print(decision.action.value, commands)
