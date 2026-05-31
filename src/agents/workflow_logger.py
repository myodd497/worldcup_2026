"""
Workflow Logger — captures execution steps from orchestrator nodes
for debugging, monitoring, and user feedback.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any


class WorkflowTracker:
    """Tracks orchestrator node execution with timestamps and payloads."""

    def __init__(self):
        self.steps: list[dict[str, Any]] = []
        self.start_time = datetime.now()

    def log_step(
        self,
        node_name: str,
        status: str = "executed",
        input_data: dict | None = None,
        output_data: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Log a single node execution.
        
        Args:
            node_name: Name of the orchestrator node
            status: "executed", "skipped", "error", etc.
            input_data: Input to the node
            output_data: Output from the node
            metadata: Additional metadata (e.g., duration, error message)
        """
        step = {
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "status": status,
            "input": input_data or {},
            "output": output_data or {},
            "metadata": metadata or {},
        }
        self.steps.append(step)

    def get_summary(self) -> str:
        """Returns a human-readable summary of workflow steps."""
        lines = ["📋 Workflow Steps:"]
        for i, step in enumerate(self.steps, 1):
            node = step["node"]
            status_emoji = "✅" if step["status"] == "executed" else "⚠️"
            lines.append(f"{i}. {status_emoji} {node} — {step['status']}")
            
            # Add key output data if available
            output = step.get("output", {})
            if "intent" in output:
                lines.append(f"   → intent: {output['intent']}")
            if "selected_agent" in output:
                lines.append(f"   → agent: {output['selected_agent']}")
            if "confidence_score" in output:
                score = output["confidence_score"]
                lines.append(f"   → confidence: {score:.0%}")
            if "answer" in output:
                answer_preview = output["answer"][:50]
                lines.append(f"   → answer: {answer_preview}...")
        
        duration = (datetime.now() - self.start_time).total_seconds()
        lines.append(f"\n⏱️  Total time: {duration:.2f}s")
        return "\n".join(lines)

    def get_json(self) -> str:
        """Returns full workflow log as JSON."""
        return json.dumps(self.steps, indent=2)

    def clear(self) -> None:
        """Clears all tracked steps."""
        self.steps = []


# Global tracker instance
_tracker: WorkflowTracker | None = None


def get_tracker() -> WorkflowTracker:
    """Get or create the global workflow tracker."""
    global _tracker
    if _tracker is None:
        _tracker = WorkflowTracker()
    return _tracker


def reset_tracker() -> None:
    """Reset the tracker for a new workflow."""
    global _tracker
    _tracker = WorkflowTracker()
