"""Deterministic, privacy-safe real-world benchmark data.

The benchmark models operational text without copying production data.  Every
value is generated from a stable per-record seed, and all hosts/identities use
reserved example namespaces.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class FamilySpec:
    domain: str
    family: str
    family_ood: bool = False


FAMILIES: Tuple[FamilySpec, ...] = (
    FamilySpec("web_api", "url"),
    FamilySpec("web_api", "http_request", True),
    FamilySpec("serialization", "json"),
    FamilySpec("serialization", "yaml_config", True),
    FamilySpec("observability", "app_log"),
    FamilySpec("observability", "stack_trace", True),
    FamilySpec("code_build", "python_code"),
    FamilySpec("code_build", "sql", True),
    FamilySpec("identifiers", "filepath"),
    FamilySpec("identifiers", "semantic_version", True),
    FamilySpec("tables_cli", "csv_table"),
    FamilySpec("tables_cli", "cli_table", True),
    FamilySpec("business", "datetime"),
    FamilySpec("business", "invoice", True),
    FamilySpec("documents", "markdown"),
    FamilySpec("documents", "chat_transcript", True),
    FamilySpec("opaque_encoding", "base64_token"),
    FamilySpec("opaque_encoding", "jwt_shaped", True),
    FamilySpec("operations", "env_file"),
    FamilySpec("operations", "cron_entry", True),
)

CONFUSABLE_FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("url", "filepath"),
    ("json", "yaml_config"),
    ("app_log", "chat_transcript"),
    ("python_code", "sql"),
    ("semantic_version", "datetime"),
    ("csv_table", "cli_table"),
    ("base64_token", "jwt_shaped"),
    ("env_file", "cron_entry"),
    ("http_request", "markdown"),
    ("invoice", "app_log"),
)


def _stable_seed(master_seed: int, *parts: object) -> int:
    payload = "|".join([str(master_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _token(rng: random.Random, length: int = 8) -> str:
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _values(rng: random.Random) -> Dict[str, object]:
    index = rng.randint(1000, 9999)
    year = rng.randint(2022, 2028)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return {
        "id": index,
        "word": _token(rng),
        "word2": _token(rng, 6),
        "year": year,
        "month": month,
        "day": day,
        "hour": rng.randint(0, 23),
        "minute": rng.randint(0, 59),
        "amount": rng.randint(100, 99999) / 100,
        "port": rng.randint(1024, 9000),
    }


def _render(family: str, template: str, rng: random.Random) -> str:
    v = _values(rng)
    i, w, w2 = v["id"], v["word"], v["word2"]
    y, m, d, hh, mm = v["year"], v["month"], v["day"], v["hour"], v["minute"]

    if family == "url":
        if template == "standard":
            return f"https://api{i}.example.invalid/v1/items/{w}?page={i % 17 + 1}&lang=en"
        if template == "alternate":
            return f"https://shop.example.invalid/products/{w2}/{i}#details"
        return f"https://[2001:db8::{i % 999:x}]:8443/a%20b/{w}?q={w2}&q={i}"
    if family == "http_request":
        if template == "standard":
            return f"GET /v1/items/{i}?view=full HTTP/1.1\nHost: api.example.invalid\nAccept: application/json"
        if template == "alternate":
            return f"POST /events HTTP/2\ncontent-type: application/json\nx-request-id: TEST-{w}\n\n{{\"id\":{i}}}"
        return f"PATCH /v2/users/{w}%2Fsettings HTTP/1.1\r\nHost: [2001:db8::{i % 99}]\r\nContent-Length: 0\r\n\r\n"
    if family == "json":
        obj = {"id": i, "name": w, "active": i % 2 == 0, "tags": [w2, "test"]}
        if template == "standard":
            return json.dumps(obj, separators=(",", ":"), sort_keys=True)
        if template == "alternate":
            return json.dumps(obj, ensure_ascii=False, indent=2)
        return json.dumps({"meta": {"request": w}, "rows": [obj, {"id": i + 1}]}, separators=(", ", ": "))
    if family == "yaml_config":
        if template == "standard":
            return f"service: {w}\nreplicas: {i % 5 + 1}\nenabled: true\nport: {v['port']}"
        if template == "alternate":
            return f"job:\n  name: {w}\n  tags:\n    - test\n    - {w2}\n  retry: {i % 4}"
        return f"defaults: &base\n  timeout: {i % 30 + 1}\nworker:\n  <<: *base\n  command: >-\n    run --id {i}"
    if family == "app_log":
        stamp = f"{y:04d}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:{i % 60:02d}Z"
        if template == "standard":
            return f"{stamp} INFO service={w} request_id=TEST-{i} duration_ms={i % 900} message=completed"
        if template == "alternate":
            return f"[{stamp}] WARN {w}.worker - retry={i % 4} queue={w2} event=delayed"
        return json.dumps({"ts": stamp, "level": "ERROR", "service": w, "trace_id": f"TEST-{w2}-{i}", "msg": "failed"})
    if family == "stack_trace":
        if template == "standard":
            return f"Traceback (most recent call last):\n  File \"/srv/{w}.py\", line {i % 300 + 1}, in run\n    process(item)\nValueError: invalid test item {i}"
        if template == "alternate":
            return f"java.lang.IllegalStateException: synthetic failure {i}\n\tat invalid.example.Service.run(Service.java:{i % 200 + 1})\n\tat invalid.example.Main.main(Main.java:12)"
        return f"Error: synthetic timeout {i}\n    at run (/srv/{w}.js:{i % 80 + 1}:7)\n    at async main (node:internal/test:1:1)\nCaused by: TEST-{w2}"
    if family == "python_code":
        if template == "standard":
            return f"def {w}(items):\n    return [item for item in items if item.id > {i % 50}]"
        if template == "alternate":
            return f"class {w.title()}:\n    def __init__(self, value: int):\n        self.value = value + {i % 10}"
        return f"async def {w}(client):\n    result = await client.get('/items/{i}')\n    return {{'ok': True, 'data': result}}"
    if family == "sql":
        if template == "standard":
            return f"SELECT id, status FROM test_{w} WHERE account_id = {i} ORDER BY id DESC LIMIT 25;"
        if template == "alternate":
            return f"INSERT INTO test_events (event_id, kind, active) VALUES ({i}, '{w2}', TRUE);"
        return f"WITH recent AS (SELECT * FROM test_{w} WHERE created_at >= DATE '{y:04d}-{m:02d}-{d:02d}') SELECT status, COUNT(*) FROM recent GROUP BY status;"
    if family == "filepath":
        if template == "standard":
            return f"/srv/app/{w}/data/{y:04d}/{m:02d}/part-{i:05d}.jsonl"
        if template == "alternate":
            return f"C:\\Users\\TestUser\\Projects\\{w}\\build\\artifact-{i}.zip"
        return f"s3://test-bucket/{w}/{y:04d}-{m:02d}-{d:02d}/{w2}%20{str(i)}.parquet"
    if family == "semantic_version":
        if template == "standard":
            return f"v{i % 12 + 1}.{i % 30}.{i % 90}"
        if template == "alternate":
            return f"{i % 9 + 1}.{i % 20}.{i % 50}-rc.{i % 7 + 1}+build.{i}"
        return f">={i % 8 + 1}.{i % 15}, <{i % 8 + 2}.0 || ~{i % 8}.{i % 10}.{i % 30}"
    if family == "csv_table":
        if template == "standard":
            return f"id,name,amount,status\n{i},{w},{v['amount']:.2f},ok\n{i + 1},{w2},{v['amount'] + 1:.2f},pending"
        if template == "alternate":
            return f'"id","note","active"\r\n"{i}","synthetic, {w}","true"\r\n"{i + 1}","line one\nline two","false"'
        return f"id\tname\tvalue\n{i}\t{w}\t{v['amount']:.2f}\n{i + 1}\t{w2}\tNULL"
    if family == "cli_table":
        if template == "standard":
            return f"NAME       READY   STATUS     AGE\n{w:<10} 1/1     Running    {i % 90}m\n{w2:<10} 0/1     Pending    {i % 12}m"
        if template == "alternate":
            return f"PID     CPU%   MEM%   COMMAND\n{i:<7} {i % 90:>4}.0  1.2    test-{w}\n{i + 1:<7} 0.1    0.4    test-{w2}"
        return f"┌──────────┬────────┬────────┐\n│ service  │ state  │ latency│\n├──────────┼────────┼────────┤\n│ {w[:8]:<8} │ ok     │ {i % 900:>4}ms │\n└──────────┴────────┴────────┘"
    if family == "datetime":
        if template == "standard":
            return f"{y:04d}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:{i % 60:02d}+00:00"
        if template == "alternate":
            return f"{d:02d}/{m:02d}/{y:04d} {hh:02d}:{mm:02d} UTC"
        return f"R{(i % 9) + 1}/{y:04d}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:00Z/P{i % 14 + 1}D"
    if family == "invoice":
        if template == "standard":
            return f"INVOICE TEST-{y}-{i}\nCustomer: Example Customer {w}\nSubtotal: USD {v['amount']:.2f}\nTax: USD {v['amount'] * .1:.2f}\nTotal: USD {v['amount'] * 1.1:.2f}"
        if template == "alternate":
            return f"Order #{i}\nItem,{w},1,{v['amount']:.2f}\nShip to: Synthetic Address {i}\nAMOUNT DUE {v['amount']:.2f} USD"
        return f"CREDIT NOTE: TEST-{i}\nOriginal: TEST-{i - 1}\nReason: synthetic return {w}\nAdjustment: ({v['amount']:.2f}) USD"
    if family == "markdown":
        if template == "standard":
            return f"# Release {i}\n\n- Added `{w}` support\n- Fixed {w2} handling\n\nSee [details](https://docs.example.invalid/{i})."
        if template == "alternate":
            return f"> Synthetic note {i}\n\n| Key | Value |\n| --- | --- |\n| name | {w} |\n| enabled | true |"
        return f"## Task {i}\n\n```json\n{{\"job\": \"{w}\", \"retry\": {i % 4}}}\n```\n\n- [x] generated\n- [ ] verified"
    if family == "chat_transcript":
        if template == "standard":
            return f"[{hh:02d}:{mm:02d}] test-user-{i}: deployed {w}\n[{hh:02d}:{(mm + 1) % 60:02d}] test-bot: build TEST-{i} passed"
        if template == "alternate":
            return f"{y:04d}-{m:02d}-{d:02d} {hh:02d}:{mm:02d} — Example User\nCan you verify {w}?\n\nExample Bot\nStatus: complete ({i})"
        return f"<test-user-{i}> {w}: hello 👋\n  ↳ <test-bot> 확인 완료 / verified\n<test-user-{i}> ref=`TEST-{w2}`"
    if family == "base64_token":
        raw = f"TEST:{i}:{w}:{w2}".encode("utf-8")
        if template == "standard":
            return base64.b64encode(raw).decode("ascii")
        if template == "alternate":
            return base64.urlsafe_b64encode(raw * 2).decode("ascii").rstrip("=")
        encoded = base64.b64encode(raw * 4).decode("ascii")
        return "\n".join(encoded[j:j + 16] for j in range(0, len(encoded), 16))
    if family == "jwt_shaped":
        header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"TEST"}').decode().rstrip("=")
        body = base64.urlsafe_b64encode(json.dumps({"sub": f"test-{i}", "aud": "example.invalid", "nonce": w}).encode()).decode().rstrip("=")
        if template == "standard":
            return f"{header}.{body}.TEST_SIGNATURE_{w2}"
        if template == "alternate":
            return f"TEST.v1.{body}.{_token(rng, 24)}"
        return f"eyJ0ZXN0Ijp0cnVlfQ.{body}.{_token(rng, 43)}"
    if family == "env_file":
        if template == "standard":
            return f"APP_NAME=test-{w}\nAPP_PORT={v['port']}\nFEATURE_ENABLED=true\nREQUEST_ID=TEST-{i}"
        if template == "alternate":
            return f"export TEST_REGION=example-{i % 4}\nexport TEST_PATH=\"/srv/{w}/data\"\nexport RETRIES='{i % 5}'"
        return f"TEST_MESSAGE=\"hello\\n{w}\"\nTEST_EMPTY=\nTEST_ESCAPED={w2}\\ value\n# synthetic {i}"
    if family == "cron_entry":
        if template == "standard":
            return f"{i % 60} {i % 23} * * {i % 7} /usr/bin/test-job --id {i} --name {w}"
        if template == "alternate":
            return f"@hourly /opt/example/{w} --output /tmp/test-{i}.log"
        return f"CRON_TZ=Etc/UTC\n*/{i % 14 + 1} {hh} {d} * 1-5 testuser /srv/{w}/run >>/var/log/test-{i}.log 2>&1"
    raise ValueError(f"Unknown family: {family}")


def generate_records(roots_per_family: int = 30, seed: int = 20260729) -> List[Dict[str, object]]:
    """Generate grouped records with leak-resistant splits and explicit OOD axes."""
    if roots_per_family < 10:
        raise ValueError("roots_per_family must be at least 10")
    records: List[Dict[str, object]] = []
    for spec in FAMILIES:
        for root_index in range(roots_per_family):
            if spec.family_ood:
                split = "family_ood"
            else:
                bucket = root_index % 10
                split = "train" if bucket < 6 else "val" if bucket < 8 else "test"
            root_id = f"{spec.family}:{root_index:04d}"
            templates = ("standard", "alternate", "ood") if split in {"test", "family_ood"} else ("standard", "alternate")
            for template in templates:
                record_seed = _stable_seed(seed, spec.family, root_index, template)
                text = _render(spec.family, template, random.Random(record_seed))
                challenge = "template_ood" if template == "ood" else split
                sample_id = hashlib.sha256(f"{root_id}|{template}|{seed}".encode()).hexdigest()[:16]
                records.append({
                    "sample_id": sample_id,
                    "root_id": root_id,
                    "domain": spec.domain,
                    "family": spec.family,
                    "template": template,
                    "split": split,
                    "challenge": challenge,
                    "family_ood": spec.family_ood,
                    "seed": record_seed,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "char_length": len(text),
                    "utf8_bytes": len(text.encode("utf-8")),
                    "text": text,
                })
    return records


def records_for(records: Sequence[Dict[str, object]], split: str, include_ood_template: bool = False) -> List[Dict[str, object]]:
    templates = {"standard", "alternate", "ood"} if include_ood_template else {"standard", "alternate"}
    return [record for record in records if record["split"] == split and record["template"] in templates]


def build_pairs(
    records: Sequence[Dict[str, object]],
    split: str,
    seed: int = 20260729,
    per_family: int = 24,
    source_template: str = "standard",
    target_template: str = "alternate",
) -> List[Dict[str, object]]:
    """Build cross-template pairs after root splitting.

    Positive pairs use the same latent root rendered through two templates.
    Negatives use the same source template and a confusable family's target
    template.  This prevents same-template OOD pairs from overstating transfer.
    """
    selected = [record for record in records if record["split"] == split]
    sources: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    targets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in selected:
        family = str(record["family"])
        if record["template"] == source_template:
            sources[family].append(record)
        if record["template"] == target_template:
            targets[family].append(record)
    rng = random.Random(_stable_seed(seed, "pairs", split, source_template, target_template, per_family))
    pairs: List[Dict[str, object]] = []
    families = sorted(set(sources) & set(targets))
    for family in families:
        source_items = sorted(sources[family], key=lambda item: str(item["sample_id"]))
        target_items = targets[family]
        target_by_root = {str(item["root_id"]): item for item in target_items}
        other_families = [name for name in families if name != family]
        if not source_items or not target_items or not other_families:
            continue
        for index in range(per_family):
            first = source_items[index % len(source_items)]
            second = target_by_root.get(str(first["root_id"]))
            if second is None:
                second = rng.choice(target_items)
            pairs.append({
                "text1": first["text"], "text2": second["text"],
                "category1": family, "category2": family, "label": 1.0,
                "difficulty": "cross_template_positive",
                "source_template": source_template, "target_template": target_template,
            })

            hard_targets = [right if left == family else left for left, right in CONFUSABLE_FAMILIES if family in (left, right) and (right if left == family else left) in targets]
            negative_family = rng.choice(hard_targets or other_families)
            negative = rng.choice(targets[negative_family])
            pairs.append({
                "text1": first["text"], "text2": negative["text"],
                "category1": family, "category2": negative_family, "label": 0.0,
                "difficulty": "hard_negative" if hard_targets else "negative",
                "source_template": source_template, "target_template": target_template,
            })
    rng.shuffle(pairs)
    return pairs


def build_training_pairs(records: Sequence[Dict[str, object]], seed: int = 20260729) -> List[Dict[str, object]]:
    return build_pairs(records, "train", seed=seed, per_family=80)


def mutate_text(text: str, mutation: str) -> str:
    """Apply recoverable operational mutations used for invariance checks."""
    if mutation == "nfd":
        return unicodedata.normalize("NFD", text)
    if mutation == "crlf":
        return text.replace("\r\n", "\n").replace("\n", "\r\n")
    if mutation == "zero_width":
        pivot = len(text) // 2
        return text[:pivot] + "\u200b" + text[pivot:]
    if mutation == "trailing_space":
        return "\n".join(line + "  " for line in text.splitlines())
    if mutation == "truncated":
        return text[:max(1, int(len(text) * 0.95))]
    raise ValueError(f"Unknown mutation: {mutation}")


def challenge_cases(records: Sequence[Dict[str, object]], limit: int = 80) -> List[Dict[str, object]]:
    base = [record for record in records if record["split"] == "test" and record["template"] == "standard"][:limit]
    output: List[Dict[str, object]] = []
    for record in base:
        for mutation in ("nfd", "crlf", "zero_width", "trailing_space", "truncated"):
            output.append({"family": record["family"], "mutation": mutation, "original": record["text"], "mutated": mutate_text(str(record["text"]), mutation)})
    return output


def boundary_cases() -> List[Dict[str, object]]:
    lengths = (0, 1, 16, 256, 4096, 100000, 100001)
    return [{"name": f"ascii_{length}", "length": length, "text": ("a,b=1|" * (length // 6 + 1))[:length]} for length in lengths]


def audit_records(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Audit determinism inputs and leakage across root-level splits."""
    roots_by_split: Dict[str, set] = defaultdict(set)
    hashes_by_split: Dict[str, set] = defaultdict(set)
    for record in records:
        roots_by_split[str(record["split"])].add(record["root_id"])
        hashes_by_split[str(record["split"])].add(record["text_sha256"])
    overlaps: Dict[str, Dict[str, int]] = {}
    names = sorted(roots_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlaps[f"{left}__{right}"] = {
                "root_overlap": len(roots_by_split[left] & roots_by_split[right]),
                "exact_text_overlap": len(hashes_by_split[left] & hashes_by_split[right]),
            }
    manifest_payload = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
    train_families = {record["family"] for record in records if record["split"] == "train"}
    held_out_families = {record["family"] for record in records if record["split"] == "family_ood"}
    train_renderers = {(record["family"], record["template"]) for record in records if record["split"] == "train"}
    template_ood_renderers = {(record["family"], record["template"]) for record in records if record["split"] == "test" and record["template"] == "ood"}
    root_and_exact_leakage_free = all(value["root_overlap"] == 0 and value["exact_text_overlap"] == 0 for value in overlaps.values())
    challenge_isolation = not (train_families & held_out_families) and not (train_renderers & template_ood_renderers)
    return {
        "records": len(records),
        "families": len({record["family"] for record in records}),
        "domains": len({record["domain"] for record in records}),
        "manifest_sha256": hashlib.sha256(manifest_payload.encode("utf-8")).hexdigest(),
        "split_counts": {name: sum(1 for record in records if record["split"] == name) for name in names},
        "overlaps": overlaps,
        "root_and_exact_leakage_free": root_and_exact_leakage_free,
        "challenge_isolation": challenge_isolation,
        "id_renderer_overlap_expected": True,
        "leakage_free": root_and_exact_leakage_free and challenge_isolation,
    }


def serializable_specs() -> List[Dict[str, object]]:
    return [asdict(spec) for spec in FAMILIES]
