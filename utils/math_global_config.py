"""Global prompt and evaluator conventions for MathGen.

The deterministic evaluators are prompt-conditioned and task-specific. These
rules document shared expectations for generated images; individual scripts may
apply stricter logic when the prompt requires it.
"""

GLOBAL_CONSTRAINTS = [
    {
        "rule": "Mathematical fidelity",
        "description": "The image must satisfy every explicit numerical, geometric, set, or functional relation in the prompt.",
    },
    {
        "rule": "No extra mathematical objects",
        "description": "Do not add extra counted objects, rays, curves, set regions, labels, or shapes that change the requested answer.",
    },
    {
        "rule": "Readable structure",
        "description": "Lines, regions, fills, labels, and axes should be visually separable enough for deterministic verification.",
    },
]

TOPIC_CONSTRAINTS = {
    "counting": [
        {
            "rule": "Exact count",
            "description": "The number of target instances must exactly match the requested quantity.",
        }
    ],
    "set": [
        {
            "rule": "Region membership",
            "description": "Only the requested Venn or set-theoretic regions should be shaded.",
        }
    ],
    "function": [
        {
            "rule": "Curve behavior",
            "description": "The plotted curve must match the requested functional behavior and visual style.",
        }
    ],
}


def get_topic_constraints(topic: str) -> list[dict]:
    """Return shared constraints for a topic, or an empty list."""
    return TOPIC_CONSTRAINTS.get(topic, [])
