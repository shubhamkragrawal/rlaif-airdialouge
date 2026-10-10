"""Command-line entry point for the local simulator."""

from __future__ import annotations

import argparse

from .controller import ConversationController


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local RLAIF flight conversation simulator")
    parser.add_argument("--adapter", default="sft")
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--model", action="store_true",
                        help="lazily load the selected local Qwen/PEFT adapter")
    args = parser.parse_args(argv)
    from .knowledge import FlightKnowledgeBase
    manager = None
    if args.model:
        from .runtime import RuntimeManager
        manager = RuntimeManager()
    controller = ConversationController(
        FlightKnowledgeBase(scenario_id=args.scenario),
        model_loader=manager.get if manager else None,
    )
    state = controller.new_session(args.adapter)
    print("Local simulated flight agent. Type 'quit' to exit.")
    while True:
        message = input("you> ").strip()
        if message.lower() in {"quit", "exit"}:
            break
        print("agent>", controller.handle(state, message)["text"])
    if manager:
        manager.release()


if __name__ == "__main__":
    main()
