
"""AURORA: Mininet-WiFi UAV-assisted transportation testbed.

Run on Linux with Mininet-WiFi installed and root privileges.
"""

import argparse
import random
from dataclasses import dataclass

from mininet.node import RemoteController
from mininet.log import setLogLevel
from mininet_wifi.net import Mininet_wifi
from mininet_wifi.node import OVSKernelAP

from aurora.acro.policy import ACROPolicyEngine, Action


@dataclass(frozen=True)
class TopologyConfig:
    num_uavs: int = 50
    num_rsus: int = 5
    controller_ip: str = "127.0.0.1"
    controller_port: int = 6653
    mobility_duration: int = 300
    max_speed: float = 20.0
    seed: int = 42


class AURORATopology:
    """UAV/RSU topology with SDN control and mobility."""

    def __init__(self, config=None):
        self.config = config or TopologyConfig()
        self.net = None
        self.uavs = {}
        self.rsus = {}
        self.policy = ACROPolicyEngine()

    def build(self):
        """Create UAV stations, RSUs, and an SDN controller."""
        cfg = self.config
        random.seed(cfg.seed)

        self.net = Mininet_wifi(
            controller=RemoteController,
            accessPoint=OVSKernelAP,
        )

        net = self.net
        net.addController(
            "c0",
            controller=RemoteController,
            ip=cfg.controller_ip,
            port=cfg.controller_port,
        )

        for i in range(cfg.num_rsus):
            x = 50 + i * 100
            ap = net.addAccessPoint(
                f"rsu{i + 1}",
                ssid=f"aurora-rsu-{i + 1}",
                mode="g",
                channel=str(1 + (i % 3) * 5),
                position=f"{x},100,0",
                range=120,
                protocols="OpenFlow13",
            )
            self.rsus[i + 1] = ap

        for i in range(cfg.num_uavs):
            x = random.randint(20, 450)
            y = random.randint(20, 180)
            uav = net.addStation(
                f"uav{i + 1}",
                ip=f"10.0.0.{i + 1}/24",
                position=f"{x},{y},20",
            )
            self.uavs[i + 1] = uav

        net.configureWifiNodes()

        # Wired backhaul connecting the RSUs.
        for i in range(1, cfg.num_rsus):
            net.addLink(
                self.rsus[i],
                self.rsus[i + 1],
            )

        return self

    def configure_mobility(self):
        """Schedule reproducible UAV movement."""
        cfg = self.config
        rng = random.Random(cfg.seed + 100)

        self.net.startMobility(time=0)

        for uav in self.uavs.values():
            x1, y1 = rng.randint(20, 450), rng.randint(20, 180)
            x2, y2 = rng.randint(20, 450), rng.randint(20, 180)

            self.net.mobility(
                uav, "start", time=1,
                position=f"{x1},{y1},20",
            )
            self.net.mobility(
                uav, "stop",
                time=cfg.mobility_duration,
                position=f"{x2},{y2},20",
            )

        self.net.stopMobility(
            time=cfg.mobility_duration + 1
        )

    def start(self):
        """Start the network and SDN connectivity."""
        if self.net is None:
            self.build()

        self.configure_mobility()
        self.net.build()
        self.net.start()

        print(
            f"AURORA started: {len(self.uavs)} UAVs, "
            f"{len(self.rsus)} RSUs"
        )
        return self

    def enforce(self, uav_id, trust):
        """Apply an ACRO decision inside the UAV namespace."""
        if uav_id not in self.uavs:
            raise KeyError(f"Unknown UAV: {uav_id}")

        decision = self.policy.evaluate(uav_id, trust)
        uav = self.uavs[uav_id]
        interface = f"{uav.name}-wlan0"

        if decision.action == Action.ISOLATE:
            uav.cmd(f"ip link set {interface} down")

        elif decision.action == Action.RATE_LIMIT:
            uav.cmd(
                f"tc qdisc replace dev {interface} root "
                "tbf rate 1000kbit burst 32kbit latency 100ms"
            )

        elif decision.action == Action.FILTER:
            uav.cmd(
                f"iptables -C OUTPUT -o {interface} -j DROP "
                f"|| iptables -I OUTPUT -o {interface} -j DROP"
            )

        return decision

    def restore(self, uav_id):
        """Restore a UAV interface and remove ACRO controls."""
        uav = self.uavs[uav_id]
        interface = f"{uav.name}-wlan0"

        uav.cmd(f"ip link set {interface} up")
        uav.cmd(
            f"tc qdisc del dev {interface} root 2>/dev/null"
        )
        uav.cmd(
            f"iptables -D OUTPUT -o {interface} -j DROP "
            "2>/dev/null"
        )
        self.policy.reset(uav_id)

    def stop(self):
        """Release all Mininet-WiFi resources."""
        if self.net is not None:
            self.net.stop()
            self.net = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uavs", type=int, default=50)
    parser.add_argument("--rsus", type=int, default=5)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--controller", default="127.0.0.1")
    args = parser.parse_args()

    cfg = TopologyConfig(
        num_uavs=args.uavs,
        num_rsus=args.rsus,
        mobility_duration=args.duration,
        controller_ip=args.controller,
    )

    topology = AURORATopology(cfg)

    try:
        topology.start()
        from mininet.cli import CLI
        CLI(topology.net)
    finally:
        topology.stop()


if __name__ == "__main__":
    setLogLevel("info")
    main()
