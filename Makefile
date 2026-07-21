.PHONY: help test golden golden-fast unit review run install hook corpus

help:
	@echo "Targets:"
	@echo "  install       create venv + install requirements"
	@echo "  run           start the dev server on :8000"
	@echo "  unit          run pytest unit tests"
	@echo "  golden        run the full golden-test suite (live LLM, ~10 min)"
	@echo "  golden-fast   only the cases tagged 'fast' (subset; not yet defined)"
	@echo "  test          unit + golden"
	@echo "  review        summarise feedback.jsonl (thumbs up/down log)"
	@echo "  hook          install a pre-commit hook that blocks commits"
	@echo "                if any golden test regresses"

install:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

run:
	./venv/bin/python run.py

unit:
	./venv/bin/python -m pytest -x

golden:
	./venv/bin/python scripts/run_golden_tests.py

review:
	./venv/bin/python scripts/review_feedback.py

# Wide-coverage auto-eval: samples real entities from DB, templates
# ~100 diverse questions across all 10 intents (plus multi-turn
# follow-ups), runs them through the live chat API with 5-way
# concurrency, applies structural assertions per intent. Output:
# pass/fail summary + starter golden-case stubs for failures.
# Server must be running. ~2-3 minutes for 100 cases.
corpus:
	./venv/bin/python scripts/generate_test_corpus.py --count 100 --concurrency 5

test: unit golden

# Installs .git/hooks/pre-commit that runs the golden suite on every
# commit. Heavy (~10 min) but it's the regression gate. Skip with
# `git commit --no-verify` when iterating fast; re-enable for landing
# commits.
hook:
	@echo '#!/usr/bin/env bash' > .git/hooks/pre-commit
	@echo 'set -e' >> .git/hooks/pre-commit
	@echo 'echo "[pre-commit] running golden suite (regression gate)..."' >> .git/hooks/pre-commit
	@echo 'cd "$$(git rev-parse --show-toplevel)"' >> .git/hooks/pre-commit
	@echo './venv/bin/python scripts/run_golden_tests.py --json --out /tmp/golden-precommit.json > /tmp/golden-precommit.log 2>&1' >> .git/hooks/pre-commit
	@echo 'fails=$$(python3 -c "import json; print(sum(1 for r in json.load(open(\"/tmp/golden-precommit.json\")) if r.get(\"pass_fail\")==\"FAIL\"))")' >> .git/hooks/pre-commit
	@echo 'if [ "$$fails" != "0" ]; then' >> .git/hooks/pre-commit
	@echo '  echo "[pre-commit] $$fails golden case(s) failing — commit blocked." ' >> .git/hooks/pre-commit
	@echo '  echo "  see /tmp/golden-precommit.log for output"' >> .git/hooks/pre-commit
	@echo '  echo "  bypass with: git commit --no-verify (use sparingly)"' >> .git/hooks/pre-commit
	@echo '  exit 1' >> .git/hooks/pre-commit
	@echo 'fi' >> .git/hooks/pre-commit
	@echo 'echo "[pre-commit] golden suite clean — commit allowed."' >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "installed .git/hooks/pre-commit (golden regression gate)"
