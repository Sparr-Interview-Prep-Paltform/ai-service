import re

from app.llm import ask_llm_json
from app.resources import get_resources
from prompts.generate_roadmap import GENERATE_ROADMAP_PROMPT


MAX_STEPS = 4
MAX_TOPICS = 3
MIN_TOPICS = 2


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sort_skills(skills):
    return sorted(
        skills,
        key=lambda skill: _to_float(skill.get("importance")),
        reverse=True,
    )


def _get_required_steps(importance):
    importance = _to_float(importance)

    if importance >= 0.8:
        return MAX_STEPS
    elif importance >= 0.5:
        return 3

    return 2


def _clean_topics(value):
    if not isinstance(value, list):
        return []

    topics = []

    for item in value:
        if not isinstance(item, str):
            continue

        topic = item.strip()

        if topic:
            topics.append(topic[:120])

        if len(topics) >= MAX_TOPICS:
            break

    return topics


def build_message(
    job_title,
    known_skills,
    skill,
    other_missing_skills=None,
):
    known = ", ".join(known_skills) if known_skills else "(none)"

    other_missing = (
        ", ".join(other_missing_skills)
        if other_missing_skills
        else "(none)"
    )

    required_steps = _get_required_steps(
        skill.get("importance")
    )

    return (
        f"<job_title>\n{job_title}\n</job_title>\n"
        f"<known_skills>\n{known}\n</known_skills>\n"
        f"<other_missing_skills>\n"
        f"{other_missing}\n"
        f"</other_missing_skills>\n"
        f"<required_steps>\n"
        f"{required_steps}\n"
        f"</required_steps>\n"
        f"<skill>\n"
        f"name: {skill['name']}\n"
        f"category: {skill.get('category', 'other')}\n"
        f"importance: {skill.get('importance')}\n"
        f"</skill>"
    )


def clean_roadmap(raw, skill, priority):
    if not isinstance(raw, dict):
        raise ValueError("Invalid roadmap response")

    raw_steps = raw.get("steps")

    if not isinstance(raw_steps, list):
        raise ValueError("Roadmap has no steps")

    required_steps = _get_required_steps(
        skill.get("importance")
    )

    steps = []

    for item in raw_steps:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()

        title = re.sub(
            r"^\s*(?:week|step)\s+\d+\s*[:\-–]\s*",
            "",
            title,
            flags=re.IGNORECASE,
        )

        title = title[:120]
        topics = _clean_topics(item.get("topics"))

        if not title or len(topics) < MIN_TOPICS:
            continue

        steps.append({
            "week": len(steps) + 1,
            "title": title,
            "topics": topics,
        })

        if len(steps) >= required_steps:
            break

    if len(steps) < required_steps:
        raise ValueError(
            f"Roadmap has {len(steps)} valid steps; "
            f"{required_steps} required"
        )

    weeks = len(steps)

    resources = [
        {
            "title": resource["title"],
            "url": resource["url"],
        }
        for resource in get_resources(skill["name"])
    ]

    return {
        "skill_id": skill.get("id"),
        "skill_name": skill["name"],
        "category": skill.get("category", "other"),
        "importance": _to_float(skill.get("importance")),
        "priority": priority,
        "title": f"Learning roadmap: {skill['name']}",
        "estimated_duration": (
            f"{weeks} week" if weeks == 1 else f"{weeks} weeks"
        ),
        "steps": steps,
        "resources": resources,
    }


def _is_complete(roadmap):
    required_steps = _get_required_steps(
        roadmap["importance"]
    )

    return len(roadmap["steps"]) == required_steps


def generate_roadmap(
    skill,
    priority,
    job_title,
    known_skills,
    other_missing_skills=None,
):
    message = build_message(
        job_title,
        known_skills,
        skill,
        other_missing_skills,
    )

    last_error = None

    for attempt in range(2):
        raw = ask_llm_json(
            GENERATE_ROADMAP_PROMPT,
            message,
        )

        try:
            roadmap = clean_roadmap(
                raw,
                skill,
                priority,
            )

            if _is_complete(roadmap):
                return roadmap

            last_error = ValueError(
                "Incomplete roadmap response"
            )

        except ValueError as error:
            last_error = error

    raise ValueError(
        f"Failed to generate a valid roadmap for skill "
        f"'{skill['name']}' after 2 attempts"
    ) from last_error