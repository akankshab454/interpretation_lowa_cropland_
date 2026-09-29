set -e
for s in steps/*.py; do
    echo "== $s"
    python -W ignore "$s"
done
