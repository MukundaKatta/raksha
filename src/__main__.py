"""CLI for raksha."""
import sys, json, argparse
from .core import Raksha

def main():
    parser = argparse.ArgumentParser(description="Raksha — AI Security Camera. Real-time threat detection and anomaly analysis from security feeds.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Raksha()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.detect(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"raksha v0.1.0 — Raksha — AI Security Camera. Real-time threat detection and anomaly analysis from security feeds.")

if __name__ == "__main__":
    main()
