.PHONY: run-daily-pipeline release release-check version

# ──────────────────────────────────────────────
# Daily Pipeline
# ──────────────────────────────────────────────
run-daily-pipeline:
	python quantpits/scripts/static_train.py --predict-only --all-enabled
	python quantpits/scripts/ensemble_fusion.py --from-config-all
	python quantpits/scripts/prod_post_trade.py
	python quantpits/scripts/order_gen.py

# ──────────────────────────────────────────────
# Release Management
# ──────────────────────────────────────────────

# Print the current version from pyproject.toml
version:
	@grep '^version' pyproject.toml | head -1

# Pre-flight checks before cutting a release
release-check:
	@echo "── Pre-release checklist ──"
	@echo "1. Tests passing?"
	@python -m pytest --tb=short -q 2>/dev/null && echo "   ✅ Tests pass" || echo "   ❌ Tests failed — fix before releasing"
	@echo "2. CHANGELOG.md updated?"
	@head -20 CHANGELOG.md | grep -q '\[Unreleased\]' && echo "   ⚠️  [Unreleased] section found — move entries to the new version section" || echo "   ✅ No stale [Unreleased] entries"
	@echo "3. pyproject.toml version:"
	@echo "   $$(grep '^version' pyproject.toml | head -1)"
	@echo "4. Current git status:"
	@git status --short

# Cut a new release: creates annotated tag, updates pyproject.toml, and pushes.
#
# Usage:
#   make release VERSION=v0.5.0-alpha
#   make release VERSION=v0.5.0-alpha MSG="My custom release message"
#
release:
	@# --- Validate inputs ---
	@test -n "$(VERSION)" || (echo "❌ Usage: make release VERSION=v0.5.0-alpha [MSG=\"...\"]" && exit 1)
	@echo $(VERSION) | grep -qE '^v[0-9]+\.[0-9]+\.[0-9]+' || (echo "❌ VERSION must start with 'v' followed by semver (e.g. v0.5.0-alpha)" && exit 1)
	@# --- Check for clean working tree ---
	@test -z "$$(git status --porcelain)" || (echo "❌ Working tree is dirty. Commit or stash changes first." && exit 1)
	@# --- Check tag doesn't already exist ---
	@git rev-parse "$(VERSION)" >/dev/null 2>&1 && (echo "❌ Tag $(VERSION) already exists!" && exit 1) || true
	@# --- Derive version string without 'v' prefix for pyproject.toml ---
	$(eval PYVER := $(shell echo $(VERSION) | sed 's/^v//'))
	@# --- Update pyproject.toml version ---
	@echo "📝 Updating pyproject.toml version to $(PYVER)..."
	@sed -i 's/^version = ".*"/version = "$(PYVER)"/' pyproject.toml
	@# --- Update __init__.py fallback version ---
	@sed -i 's/__version__ = ".*"/__version__ = "$(PYVER)"/' quantpits/__init__.py
	@# --- Commit version bump ---
	@git add pyproject.toml quantpits/__init__.py
	@git commit -m "chore(release): bump version to $(VERSION)"
	@# --- Create annotated tag ---
	@echo "🏷️  Creating annotated tag $(VERSION)..."
	@git tag -a $(VERSION) -m "$${MSG:-Release $(VERSION)}"
	@# --- Push ---
	@echo "🚀 Pushing commit and tag to origin..."
	@git push origin main
	@git push origin $(VERSION)
	@echo ""
	@echo "✅ Release $(VERSION) complete!"
	@echo ""
	@echo "📋 Next steps:"
	@echo "   1. Go to https://github.com/DarkLink/QuantPits/releases/new?tag=$(VERSION)"
	@echo "   2. Copy the relevant section from CHANGELOG.md as the release body"
	@echo "   3. Publish the release"
