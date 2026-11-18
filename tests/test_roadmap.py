import pytest

from app import roadmap


def test_sort_skills_by_importance():
    skills = [
        {"name": "SQL", "importance": 0.5},
        {"name": "Django", "importance": 0.9},
        {"name": "Docker", "importance": 0.7},
    ]

    result = roadmap.sort_skills(skills)

    assert [skill["name"] for skill in result] == [
        "Django",
        "Docker",
        "SQL",
    ]


def test_sort_skills_invalid_importance():
    skills = [
        {"name": "Python", "importance": "invalid"},
        {"name": "Django", "importance": 0.8},
        {"name": "SQL", "importance": None},
    ]

    result = roadmap.sort_skills(skills)

    assert result[0]["name"] == "Django"


def test_get_required_steps():
    assert roadmap._get_required_steps(0.9) == 4
    assert roadmap._get_required_steps(0.8) == 4
    assert roadmap._get_required_steps(0.75) == 3
    assert roadmap._get_required_steps(0.5) == 3
    assert roadmap._get_required_steps(0.4) == 2


def test_clean_topics_limits_to_three():
    topics = [
        "Topic 1",
        "Topic 2",
        "Topic 3",
        "Topic 4",
        "Topic 5",
    ]

    result = roadmap._clean_topics(topics)

    assert len(result) == 3
    assert result == ["Topic 1", "Topic 2", "Topic 3"]


def test_clean_topics_invalid_input():
    assert roadmap._clean_topics("Python") == []
    assert roadmap._clean_topics(None) == []


def test_clean_topics_removes_empty_values():
    result = roadmap._clean_topics([
        " Python ",
        "",
        " Django ",
        "   ",
    ])

    assert result == ["Python", "Django"]


def test_clean_topics_ignores_non_string_values():
    result = roadmap._clean_topics([
        "Python",
        None,
        123,
        "Django",
    ])

    assert result == ["Python", "Django"]


def test_build_message_includes_required_steps():
    message = roadmap.build_message(
        "Backend Developer",
        ["Python", "SQL"],
        {
            "name": "Django",
            "category": "frameworks",
            "importance": 0.95,
        },
    )

    assert "<required_steps>\n4\n</required_steps>" in message
    assert "Backend Developer" in message
    assert "Django" in message


def test_clean_roadmap_success(monkeypatch):
    monkeypatch.setattr(
        roadmap,
        "get_resources",
        lambda skill_name: [
            {
                "title": "Official documentation",
                "url": "https://example.com",
            }
        ],
    )

    skill = {
        "id": 1,
        "name": "Django",
        "category": "frameworks",
        "importance": 0.9,
    }

    raw = {
        "steps": [
            {
                "title": "Django fundamentals",
                "topics": ["Models", "Views", "URLs"],
            },
            {
                "title": "Database management",
                "topics": ["ORM", "Migrations", "Admin"],
            },
            {
                "title": "Build application features",
                "topics": ["Forms", "Templates", "Validation"],
            },
            {
                "title": "Build a Django project",
                "topics": ["Project setup", "Testing", "Documentation"],
            },
        ]
    }

    result = roadmap.clean_roadmap(raw, skill, 1)

    assert result["skill_name"] == "Django"
    assert result["skill_id"] == 1
    assert result["priority"] == 1
    assert result["importance"] == 0.9
    assert result["estimated_duration"] == "4 weeks"
    assert len(result["steps"]) == 4
    assert result["steps"][0]["week"] == 1
    assert result["steps"][-1]["week"] == 4
    assert result["resources"][0]["title"] == "Official documentation"


def test_clean_roadmap_limits_steps(monkeypatch):
    monkeypatch.setattr(
        roadmap,
        "get_resources",
        lambda skill_name: [],
    )

    skill = {
        "name": "Python",
        "importance": 0.8,
    }

    raw = {
        "steps": [
            {
                "title": f"Step {i}",
                "topics": ["Topic A", "Topic B", "Topic C"],
            }
            for i in range(1, 9)
        ]
    }

    result = roadmap.clean_roadmap(raw, skill, 1)

    assert len(result["steps"]) == 4
    assert result["steps"][-1]["week"] == 4


def test_clean_roadmap_generates_three_steps_for_medium_importance(
    monkeypatch,
):
    monkeypatch.setattr(
        roadmap,
        "get_resources",
        lambda skill_name: [],
    )

    skill = {
        "name": "PostgreSQL",
        "importance": 0.75,
    }

    raw = {
        "steps": [
            {
                "title": f"Step {i}",
                "topics": ["Topic A", "Topic B"],
            }
            for i in range(1, 5)
        ]
    }

    result = roadmap.clean_roadmap(raw, skill, 1)

    assert len(result["steps"]) == 3
    assert result["estimated_duration"] == "3 weeks"


def test_clean_roadmap_generates_two_steps_for_low_importance(
    monkeypatch,
):
    monkeypatch.setattr(
        roadmap,
        "get_resources",
        lambda skill_name: [],
    )

    skill = {
        "name": "Communication",
        "importance": 0.4,
    }

    raw = {
        "steps": [
            {
                "title": f"Step {i}",
                "topics": ["Topic A", "Topic B"],
            }
            for i in range(1, 4)
        ]
    }

    result = roadmap.clean_roadmap(raw, skill, 1)

    assert len(result["steps"]) == 2
    assert result["estimated_duration"] == "2 weeks"


def test_clean_roadmap_skips_invalid_steps(monkeypatch):
    monkeypatch.setattr(
        roadmap,
        "get_resources",
        lambda skill_name: [],
    )

    skill = {
        "name": "Python",
        "importance": 0.4,
    }

    raw = {
        "steps": [
            None,
            {"title": "", "topics": []},
            {
                "title": "Step 1: Python basics",
                "topics": ["Basic concepts", "Core principles"],
            },
            {
                "title": "Step 2: Practical Python",
                "topics": ["Practical examples", "Exercises"],
            },
        ]
    }

    result = roadmap.clean_roadmap(raw, skill, 1)

    assert len(result["steps"]) == 2
    assert result["steps"][0]["week"] == 1
    assert result["steps"][0]["title"] == "Python basics"
    assert result["steps"][1]["week"] == 2


def test_clean_roadmap_missing_steps():
    with pytest.raises(ValueError, match="no steps"):
        roadmap.clean_roadmap(
            {},
            {"name": "Python"},
            1,
        )


def test_clean_roadmap_insufficient_valid_steps(monkeypatch):
    monkeypatch.setattr(
        roadmap,
        "get_resources",
        lambda skill_name: [],
    )

    with pytest.raises(ValueError, match="valid steps"):
        roadmap.clean_roadmap(
            {
                "steps": [
                    {
                        "title": "Python basics",
                        "topics": ["Variables", "Data types"],
                    }
                ]
            },
            {
                "name": "Python",
                "importance": 0.9,
            },
            1,
        )


def test_clean_roadmap_invalid_response():
    with pytest.raises(ValueError, match="Invalid roadmap response"):
        roadmap.clean_roadmap(
            None,
            {"name": "Python"},
            1,
        )


def test_clean_roadmap_requires_two_topics(monkeypatch):
    monkeypatch.setattr(
        roadmap,
        "get_resources",
        lambda skill_name: [],
    )

    with pytest.raises(ValueError, match="valid steps"):
        roadmap.clean_roadmap(
            {
                "steps": [
                    {
                        "title": "Python basics",
                        "topics": ["Variables"],
                    },
                    {
                        "title": "Python practice",
                        "topics": ["Exercises"],
                    },
                ]
            },
            {
                "name": "Python",
                "importance": 0.4,
            },
            1,
        )


def test_generate_roadmap_calls_llm_and_cleans(monkeypatch):
    raw = {
        "steps": [
            {
                "title": "Learn Python fundamentals",
                "topics": [
                    "Variables and data types",
                    "Control flow",
                    "Functions",
                ],
            },
            {
                "title": "Work with Python data structures",
                "topics": [
                    "Lists and tuples",
                    "Dictionaries and sets",
                    "Data manipulation",
                ],
            },
            {
                "title": "Apply Python concepts",
                "topics": [
                    "Error handling",
                    "File operations",
                    "Code organization",
                ],
            },
            {
                "title": "Build a practical Python project",
                "topics": [
                    "Project planning",
                    "Implementation",
                    "Testing and debugging",
                ],
            },
        ]
    }

    monkeypatch.setattr(
        roadmap,
        "ask_llm_json",
        lambda prompt, message: raw,
    )

    monkeypatch.setattr(
        roadmap,
        "get_resources",
        lambda skill_name: [],
    )

    skill = {
        "id": 5,
        "name": "Python",
        "category": "languages",
        "importance": 0.95,
    }

    result = roadmap.generate_roadmap(
        skill,
        1,
        "Backend Developer",
        ["SQL", "Git"],
    )

    assert result["skill_name"] == "Python"
    assert result["priority"] == 1
    assert len(result["steps"]) == 4
    assert result["steps"][-1]["title"] == (
        "Build a practical Python project"
    )


def test_generate_roadmap_retries_after_invalid_response(monkeypatch):
    invalid_response = {
        "steps": [
            {
                "title": "Only one step",
                "topics": ["Topic A", "Topic B"],
            }
        ]
    }

    valid_response = {
        "steps": [
            {
                "title": f"Step {i}",
                "topics": ["Topic A", "Topic B"],
            }
            for i in range(1, 5)
        ]
    }

    responses = iter([invalid_response, valid_response])
    calls = {"count": 0}

    def mock_ask_llm_json(prompt, message):
        calls["count"] += 1
        return next(responses)

    monkeypatch.setattr(
        roadmap,
        "ask_llm_json",
        mock_ask_llm_json,
    )

    monkeypatch.setattr(
        roadmap,
        "get_resources",
        lambda skill_name: [],
    )

    skill = {
        "name": "Python",
        "importance": 0.9,
    }

    result = roadmap.generate_roadmap(
        skill,
        1,
        "Backend Developer",
        ["SQL"],
    )

    assert calls["count"] == 2
    assert len(result["steps"]) == 4


def test_generate_roadmap_fails_after_two_invalid_responses(
    monkeypatch,
):
    calls = {"count": 0}

    def mock_ask_llm_json(prompt, message):
        calls["count"] += 1
        return {"steps": []}

    monkeypatch.setattr(
        roadmap,
        "ask_llm_json",
        mock_ask_llm_json,
    )

    skill = {
        "name": "Python",
        "importance": 0.9,
    }

    with pytest.raises(ValueError, match="after 2 attempts"):
        roadmap.generate_roadmap(
            skill,
            1,
            "Backend Developer",
            [],
        )

    assert calls["count"] == 2