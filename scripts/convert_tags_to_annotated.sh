#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# convert_tags_to_annotated.sh
#
# Converts all existing lightweight tags to annotated tags,
# preserving the original commit they point to and embedding
# the original commit message + date as the tag message.
#
# This is a LOCAL-ONLY operation. You must force-push tags
# to remote after review:
#   git push origin --tags --force
# ────────────────────────────────────────────────────────────────

set -euo pipefail

echo "🔄 Converting lightweight tags to annotated tags..."
echo ""

# Use the committer's identity for tagging
export GIT_COMMITTER_NAME="$(git config user.name)"
export GIT_COMMITTER_EMAIL="$(git config user.email)"

TAGS=$(git tag -l --sort=version:refname)

for TAG in $TAGS; do
    # Check if it's already annotated (type = tag) vs lightweight (type = commit)
    OBJ_TYPE=$(git cat-file -t "$TAG")

    if [ "$OBJ_TYPE" = "tag" ]; then
        echo "  ⏭️  $TAG — already annotated, skipping"
        continue
    fi

    # Get the commit this lightweight tag points to
    COMMIT=$(git rev-list -n1 "$TAG")

    # Get the original commit date and message for the tag message
    COMMIT_DATE=$(git log -1 --format="%ai" "$COMMIT")
    COMMIT_MSG=$(git log -1 --format="%B" "$COMMIT")

    # Preserve original commit date as the tagger date
    export GIT_COMMITTER_DATE="$COMMIT_DATE"

    echo "  🏷️  $TAG → annotated (commit: ${COMMIT:0:7}, date: ${COMMIT_DATE:0:10})"

    # Delete the lightweight tag
    git tag -d "$TAG" > /dev/null 2>&1

    # Recreate as annotated tag pointing to the same commit
    git tag -a "$TAG" "$COMMIT" -m "Release $TAG

$COMMIT_MSG"
done

unset GIT_COMMITTER_DATE

echo ""
echo "✅ All tags converted to annotated tags."
echo ""
echo "To verify:"
echo "  git tag -l | xargs -I{} git cat-file -t {}"
echo ""
echo "To push to remote (requires force-push):"
echo "  git push origin --tags --force"
