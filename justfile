default:
    @just --list

check:
    ruff check rubrify/ tests/
    ruff format --check rubrify/ tests/
    mypy rubrify/

format:
    ruff format rubrify/ tests/
    ruff check --fix rubrify/ tests/

test *args:
    pytest tests/ -m "not integration" {{args}}

test-all *args:
    pytest tests/ {{args}}

build:
    python -m build
