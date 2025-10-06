#!/bin/bash
set -e

# Script to upload coverage to Codecov
# Usage: upload_codecov.sh "Step Label"

STEP_LABEL="${1:-unknown}"

# Convert step label to flag format (lowercase, replace special chars with underscores)
FLAG=$(echo "$STEP_LABEL" | tr '[:upper:]' '[:lower:]' | sed 's/[() %,+]/_/g' | sed 's/__*/_/g' | sed 's/^_//;s/_$//')

# Check if codecov token and coverage.xml exist
if [ -z "${CODECOV_TOKEN:-}" ]; then
    echo "CODECOV_TOKEN not set, skipping upload"
    exit 0
fi

if [ ! -f coverage.xml ]; then
    echo "coverage.xml not found in $(pwd), skipping upload"
    exit 0
fi

echo "Found coverage.xml in $(pwd)"
echo "Sample paths before normalization:"
grep 'filename=' coverage.xml | head -3 || true

# Normalize coverage database files as well, so if Codecov runs combine/xml
# it will regenerate coverage.xml with canonical paths.
python3 - <<'PY' || true
import glob, os, sqlite3, sys

def normalize_db(db_path: str) -> None:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        for table in ("file", "files"):
            try:
                cur.execute(f"PRAGMA table_info({table})")
                cols = [c[1] for c in cur.fetchall()]
                if "path" not in cols:
                    continue
                # Any .../(site|dist)-packages/vllm/... -> vllm/...
                cur.execute(
                    f"""
                    UPDATE {table}
                    SET path = 'vllm/' || SUBSTR(path, INSTR(path, '/vllm/')+6)
                    WHERE path LIKE '%/site-packages/vllm/%' OR path LIKE '%/dist-packages/vllm/%'
                    """
                )
                # /vllm-workspace/vllm/... -> vllm/...
                cur.execute(
                    f"""
                    UPDATE {table}
                    SET path = 'vllm/' || SUBSTR(path, INSTR(path, '/vllm/')+6)
                    WHERE path LIKE '/vllm-workspace/vllm/%'
                    """
                )
                # ./vllm/... and ../vllm/... -> vllm/...
                cur.execute(
                    f"""
                    UPDATE {table}
                    SET path = 'vllm/' || SUBSTR(path, INSTR(path, '/vllm/')+6)
                    WHERE path LIKE './vllm/%' OR path LIKE '../vllm/%'
                    """
                )
                conn.commit()
            except Exception:
                continue
    except Exception as e:
        print(f"[coverage-db-normalize] {db_path}: {e}", file=sys.stderr)
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass

for db in glob.glob('.coverage*'):
    if os.path.isfile(db):
        normalize_db(db)
PY

# Normalize filenames in coverage.xml to ensure consistent paths across uploads
# Map any site/dist-packages and workspace-relative paths to canonical "vllm/"
# Works across /usr, /usr/local, /opt/conda, virtualenvs, etc.
if sed -i \
    -e 's@filename="[^"]*/site-packages/vllm/@filename="vllm/@g' \
    -e 's@filename="[^"]*/dist-packages/vllm/@filename="vllm/@g' \
    -e 's@filename="/vllm-workspace/vllm/@filename="vllm/@g' \
    -e 's@filename="\./vllm/@filename="vllm/@g' \
    -e 's@filename="\.\./vllm/@filename="vllm/@g' \
    coverage.xml 2>/dev/null; then
    echo "✓ Path normalization successful"
else
    echo "⚠ Warning: sed path normalization failed, uploading coverage as-is"
fi

echo "Sample paths after normalization:"
grep 'filename=' coverage.xml | head -3 || true

# Download codecov CLI if not present
if [ ! -f codecov ]; then
    curl -Os https://cli.codecov.io/latest/linux/codecov
    chmod +x codecov
fi

# Upload to codecov
./codecov upload-process \
    -t "${CODECOV_TOKEN}" \
    -f coverage.xml \
    --git-service github \
    --build "${BUILDKITE_BUILD_NUMBER:-unknown}" \
    --branch "${BUILDKITE_BRANCH:-unknown}" \
    --sha "${BUILDKITE_COMMIT:-unknown}" \
    --slug vllm-project/vllm \
    --flag "$FLAG" \
    --name "$STEP_LABEL" \
    --dir /vllm-workspace || true

exit 0
