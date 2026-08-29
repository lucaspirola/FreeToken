from __future__ import annotations

import sys
from pathlib import Path


BENCH = Path(__file__).parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH))
import bench_multi_agent_context as bench  # noqa: E402


def test_synthetic_agents_have_unique_needles_and_prefixes():
    samples = [bench.synthetic_agent_sample(agent) for agent in range(4)]
    prompts = [sample[0] for sample in samples]
    needles = [sample[1] for sample in samples]

    assert len(set(needles)) == 4
    assert len({prompt[:256] for prompt in prompts}) == 4
    for agent, (prompt, needle) in enumerate(samples):
        assert needle in prompt
        assert all(other not in prompt for other in needles if other != needle)
        assert f"isolated agent {agent}" in prompt
