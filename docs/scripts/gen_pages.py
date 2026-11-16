"""Generate the scenario pages, their animations and the benchmark figures for the docs site.

Run by mkdocs-gen-files during `mkdocs build` / `mkdocs serve`. Every table is built from the
registered scenario configurations, so the pages always match the code.
"""

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import mkdocs_gen_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from animation import scenario_gif  # noqa: E402

import gridhunt  # noqa: E402
from gridhunt import zoo  # noqa: E402

SUMMARY = ROOT / "benchmark" / "results" / "summary"

SCENARIOS = {
    "classic_stag": {
        "env_id": "GridHunt-ClassicStag-v0",
        "title": "Classic stag hunt",
        "summary": (
            "The two-player stag hunt on a 5×5 grid. The stag is worth 5 but needs both hunters on its "
            "cell at once; a hare is worth 1 and one hunter can catch it. Captured prey reappear on a "
            "random free cell. With individual rewards it is a social dilemma: hares are safe, the stag "
            "pays more but only if the partner commits too."
        ),
    },
    "toxic_stag": {
        "env_id": "GridHunt-ToxicStag-v0",
        "title": "Toxic stag",
        "summary": (
            "Three hunters on a 7×7 grid. Every time the stag appears it is toxic with probability 0.5: a "
            "healthy stag is worth 5 and a toxic one costs 5, and a toxic stag left alone for 5 steps "
            "reappears elsewhere with its toxicity drawn again. Only `predator_0` can tell whether the "
            "stag is toxic, and it sees only the cells next to it, so the team has to act on one member's "
            "private information."
        ),
    },
    "specialists": {
        "env_id": "GridHunt-Specialists-v0",
        "title": "Specialists",
        "summary": (
            "A scout and a bruiser on a 10×10 grid. The scout is fast and sees the whole board but is too "
            "weak to take the stag; the bruiser can take the stag on its own but sees only two cells "
            "around it. Prey flee from the nearest hunter, so the bruiser depends on what the scout knows: "
            "an information-interdependence task."
        ),
    },
    "military": {
        "env_id": "GridHunt-Military-v0",
        "title": "Military",
        "summary": (
            "A UAV and a ground vehicle on a 10×10 grid with mountains, swamps and impassable walls that "
            "are generated anew every episode. The UAV flies over all terrain and sees the whole board but "
            "cannot capture anything; the ground vehicle does all the capturing but is slowed by terrain "
            "and sees only three cells around it."
        ),
    },
    "raid": {
        "env_id": "GridHunt-Raid-v0",
        "title": "Raid",
        "summary": (
            "Three raiders on a 12×12 grid with a barn, a house and a bank. Richer targets need more "
            "raiders at once and are more likely to be trapped, and a trap costs the whole team. Each "
            "building can be raided once per episode."
        ),
    },
    "pig_chase": {
        "env_id": "GridHunt-PigChase-v0",
        "title": "Pig chase",
        "summary": (
            "Two hunters start in opposite corners of a 9×9 maze with the pig in the middle. Catching the "
            "pig needs both hunters on its cell at once; after every capture everyone returns to their "
            "starting cells. The scenario has no randomness, so it tests whether a team can learn and keep "
            "a fixed joint plan."
        ),
    },
    "land_and_sea": {
        "env_id": "GridHunt-LandAndSea-v0",
        "title": "Land and sea",
        "summary": (
            "A 10×10 grid split into sea and land along an irregular shore. One hunter is confined to land "
            "and the other to sea; either can catch the bird, which reappears on a random cell after every "
            "capture, so the hunter on the right side of the shore has to go for it."
        ),
    },
    "swarm": {
        "env_id": "GridHunt-Swarm-v0",
        "title": "Swarm",
        "summary": (
            "Ten hunters on a 23×23 grid with three stags and five hares. Stags need two hunters, and more "
            "than one hunter on a hare costs half its value, so the team has to spread out and pair up."
        ),
    },
}


def _vision(predator):
    if predator.status.get("limited_vision"):
        radius = predator.limited_vision_radius
        return f"{radius} cell{'s' if radius != 1 else ''} around it"
    return "whole board"


def _terrain_rules(predator):
    if predator.immune_to_terrain:
        return "ignores terrain"
    if predator.restricted_terrains:
        return "cannot enter " + ", ".join(sorted(predator.restricted_terrains))
    return "-"


def _prey_notes(prey):
    notes = []
    if hasattr(prey, "toxic_probability"):
        notes.append(f"toxic with probability {prey.toxic_probability} (then worth -{abs(prey.reward)})")
    if prey.status.get("trap_probability"):
        notes.append(f"trapped with probability {prey.status['trap_probability']} "
                     f"(team penalty {prey.status['trap_penalty']})")
    if prey.overcrowding_penalty:
        notes.append(f"overcrowding costs {prey.overcrowding_penalty}")
    if prey.speed > 1:
        notes.append(f"speed {prey.speed}")
    return "; ".join(notes) or "-"


def _terrain_rows(config):
    rows = []
    for name, settings in config.items():
        generation = settings.get("generation", {})
        kind = generation.get("type", "blob")
        layout = {"predefined": "fixed layout", "blob": "random blobs each episode",
                  "line": "random walls each episode"}.get(kind, kind)
        passability = settings.get("passability", 1.0)
        entry = "impassable" if passability == 0 else ("passable" if passability == 1 else
                                                       f"each move in succeeds with probability {passability}")
        rows.append(f"| {name} | {entry} | {layout} |")
    return rows


def _benchmark_rows(name):
    path = SUMMARY / "final_scores.csv"
    if not path.exists():
        return []
    configurations = {name} | {f"{name}_{variant}" for variant in ("comm", "full_vision", "blind")}
    out = []
    for r in csv.DictReader(path.open()):
        if r["scenario"] not in configurations:
            continue
        out.append(f"| `{r['scenario']}` | {r['algo'].upper()} | {float(r['iqm']):.1f} "
                   f"[{float(r['iqm_lo']):.1f}, {float(r['iqm_hi']):.1f}] | {float(r['norm_iqm']):.2f} | "
                   f"{float(r['random']):.1f} | {float(r['scripted']):.1f} |")
    return out


def scenario_page(name, info, protocol):
    env_id = info["env_id"]
    config = gridhunt._BENCHMARKS[env_id]
    env = gridhunt.make(env_id)
    agents = env.possible_agents
    obs_sizes = sorted({env.observation_space(a).shape[0] for a in agents})
    lines = [
        f"# {info['title']}",
        "",
        f"![{info['title']}](../assets/scenarios/{name}.gif){{ width=\"360\" }}",
        "",
        info["summary"],
        "",
        "```python",
        "import gridhunt",
        f'env = gridhunt.make("{env_id}")',
        "```",
        "",
        "| | |",
        "|---|---|",
        f"| Environment ID | `{env_id}` |",
        f"| Grid | {config['grid_size'][0]} × {config['grid_size'][1]} |",
        f"| Agents | {len(agents)} |",
        f"| Episode length | {config['max_steps']} steps |",
        "| Actions | `Discrete(5)`: stay, up, down, left, right |",
        f"| Observations | `{config['observation_mode']}`, {' or '.join(map(str, obs_sizes))} values per agent |",
        f"| Global state | {env.state_space.shape[0]} values (`env.state()`) |",
        f"| Prey motion | {config.get('prey_motion', 'static')}"
        + (f" (each step with probability {config['prey_move_probability']})"
           if config.get("prey_motion", "static") != "static" else "") + " |",
        "",
        "## Agents",
        "",
        "| Agent | Capture power | Speed | Vision | Terrain | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for predator in config["predators"]:
        notes = []
        if predator.status.get("toxicity_aware"):
            notes.append("knows whether the stag is toxic")
        if predator.position is not None:
            notes.append(f"starts at {predator.position}")
        lines.append(f"| `{predator.predator_id}` | {predator.capture_power} | {predator.speed} | "
                     f"{_vision(predator)} | {_terrain_rules(predator)} | {'; '.join(notes) or '-'} |")
    lines += [
        "",
        "## Prey",
        "",
        "| Prey | Type | Reward | Capture power needed | Reappears after capture | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for prey in config["prey"]:
        lines.append(f"| `{prey.prey_id}` | {prey.type} | {prey.reward} | "
                     f"{prey.capture_power_required} | {'yes' if prey.respawn else 'no'} | {_prey_notes(prey)} |")
    lines += [
        "",
        "A capture happens when the hunters on a prey's cell have at least the capture power it needs. "
        "With more power than needed, the prey's overcrowding penalty is subtracted from its reward.",
    ]
    if config.get("terrain_config"):
        lines += ["", "## Terrain", "", "| Terrain | Movement | Layout |", "|---|---|---|",
                  *_terrain_rows(config["terrain_config"])]
    if protocol:
        spec = protocol["spec"]
        lines += [
            "",
            "## Benchmark",
            "",
            f"The benchmark plays it with team reward, {spec['max_steps']}-step episodes and a budget of "
            f"{spec['budget'] / 1e6:g}M environment steps"
            + (f", with `{spec['env_kwargs']['observation_mode']}` observations"
               if spec.get("env_kwargs", {}).get("observation_mode") else "")
            + ". Scores are team returns: IQM over 20 seeds with 95% bootstrap intervals "
            "([details](../benchmark/index.md)).",
            "",
            "| Configuration | Algorithm | IQM [95% CI] | Normalized | Random | Scripted |",
            "|---|---|---|---|---|---|",
            *_benchmark_rows(name),
        ]
    lines += ["", "## Version history", "", "- `v0` (GridHunt 1.0.0): first release.", ""]
    return "\n".join(lines)


def main():
    import protocol

    index = [
        "# Environments",
        "",
        "Every scenario is a PettingZoo `ParallelEnv` created with `gridhunt.make(env_id)`. Any setting "
        "can be overridden there, for example `max_steps`, `reward_mode=\"team\"` or "
        "`observation_mode`; see [custom scenarios](../guides/custom-scenarios.md) to build your own.",
        "",
    ]
    table = ["| Scenario | Environment ID | Grid | Agents | Challenge |", "|---|---|---|---|---|"]
    for name, info in SCENARIOS.items():
        spec = protocol.SCENARIOS.get(name)
        page = scenario_page(name, info, {"spec": spec} if spec else None)
        with mkdocs_gen_files.open(f"environments/{name}.md", "w") as f:
            f.write(page)
        mkdocs_gen_files.set_edit_path(f"environments/{name}.md", "../scripts/gen_pages.py")
        with mkdocs_gen_files.open(f"assets/scenarios/{name}.gif", "wb") as f:
            f.write(scenario_gif(info["env_id"]))
        config = gridhunt._BENCHMARKS[info["env_id"]]
        first_sentence = info["summary"].split(". ")[0].rstrip(".") + "."
        table.append(f"| [{info['title']}]({name}.md) | `{info['env_id']}` | {config['grid_size'][0]}×"
                     f"{config['grid_size'][1]} | {len(config['predators'])} | {first_sentence} |")
    index += table + ["", "Each page lists the agents, prey and terrain exactly as registered, and the "
                      "benchmark results for that scenario.", ""]
    with mkdocs_gen_files.open("environments/index.md", "w") as f:
        f.write("\n".join(index))

    # Benchmark figures
    for path in sorted((SUMMARY / "curves").glob("*.png")) + [SUMMARY / "aggregate.png"]:
        if path.exists():
            target = "benchmark/curves/" + path.name if path.parent.name == "curves" else "benchmark/" + path.name
            with mkdocs_gen_files.open(target, "wb") as f:
                f.write(path.read_bytes())

    # Published pretrained teams
    with mkdocs_gen_files.open("benchmark/zoo.md", "w") as f:
        f.write(zoo_page())
    mkdocs_gen_files.set_edit_path("benchmark/zoo.md", "../scripts/gen_pages.py")


def zoo_page():
    models = zoo.list_models()
    lines = [
        "# Pretrained teams",
        "",
        "`gridhunt.zoo` provides the benchmark's trained teams as numpy-only policies: no PyTorch is "
        "needed to run them. For every scenario and algorithm two runs are published: the seed with "
        "the median final score (typical of the algorithm) and the best seed (the strongest teammates).",
        "",
        "```python",
        "from gridhunt import zoo",
        "",
        'team = zoo.load("toxic_stag", "mappo", rank="best")  # downloaded once, then cached',
        "env = team.make_env()                                # the scenario exactly as trained",
        "observations, _ = env.reset(seed=0)",
        "while env.agents:",
        "    observations, *_ = env.step(team.act(observations))",
        "```",
        "",
        "- `team.act(observations)` samples actions, as the benchmark evaluates; pass `greedy=True` for "
        "the most likely action. `zoo.load(..., rng=seed)` makes sampling reproducible.",
        "- `team.make_env()` rebuilds the benchmark configuration of the scenario, including the "
        "variants; overriding `max_steps` or `render_mode` is fine, observation settings are not.",
        "- A team is also a teammate for a human study: `HumanSession(env, humans=..., "
        "teammates={\"predator_1\": team})` (see [sessions](../hat/sessions.md)).",
        "- Files are checked against their SHA-256 hash and cached in `~/.cache/gridhunt/zoo` "
        "(`$GRIDHUNT_ZOO_DIR` to change it, `$GRIDHUNT_ZOO_URL` for a mirror).",
        "",
        "Before publishing, every exported team is checked against the PyTorch model it came from: equal "
        "action logits on observations from real episodes, greedy actions that reproduce the recorded "
        "greedy evaluation episode by episode, and a sampled score within 4 standard errors of the "
        "recorded one ([`benchmark/export_zoo.py`](https://github.com/ajuanijustus/gridhunt/blob/main/"
        "benchmark/export_zoo.py)).",
        "",
        "## Published teams",
        "",
    ]
    if not models:
        lines += ["!!! note", "    No pretrained teams are published with this version yet.", ""]
    else:
        lines += ["| Scenario | Algorithm | Rank | Seed | Score (sampled) | Size |", "|---|---|---|---|---|---|"]
        for m in models:
            lines.append(f"| `{m['scenario']}` | {m['algo'].upper()} | {m['rank'] or '-'} | {m['seed']} | "
                         f"{m['sampled_return_mean']:.1f} | {m.get('size_bytes', 0) / 1e6:.1f} MB |")
        lines.append("")
    return "\n".join(lines)


main()
