"""Create the log index in OpenSearch and load the demo log lines (timestamps relative to now).

.venv/bin/python scripts/seed_opensearch.py --url http://localhost:9200 --index app-logs
"""

from __future__ import annotations

import argparse
import sys

from relay.domain import utcnow
from relay.tools.backends import load_demo_logs

KEYWORD_TEXT = {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}

MAPPING = {
    "mappings": {
        "properties": {
            "@timestamp": {"type": "date"},
            "service": KEYWORD_TEXT,
            "level": KEYWORD_TEXT,
            "customer_id": KEYWORD_TEXT,
            "request_id": KEYWORD_TEXT,
            "status_code": {"type": "integer"},
            "message": {"type": "text"},
        }
    }
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:9200")
    parser.add_argument("--index", default="app-logs")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args(argv)
    try:
        from opensearchpy import OpenSearch, helpers
    except ImportError:
        print("install the opensearch extra: pip install 'relay[opensearch]'", file=sys.stderr)
        return 1

    client = OpenSearch(hosts=[args.url])
    if args.recreate and client.indices.exists(index=args.index):
        client.indices.delete(index=args.index)
    if not client.indices.exists(index=args.index):
        client.indices.create(index=args.index, body=MAPPING)
    actions = []
    for row in load_demo_logs(utcnow()):
        doc = dict(row)
        doc["@timestamp"] = doc.pop("timestamp")
        actions.append({"_index": args.index, "_source": doc})
    helpers.bulk(client, actions, refresh=True)
    print(f"indexed {len(actions)} log lines into {args.index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
