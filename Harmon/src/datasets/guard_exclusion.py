"""Fail-closed training exclusions for GUARD calibration image groups."""

import hashlib
import json
import threading
import unicodedata
from pathlib import Path


class GuardExclusionError(ValueError):
    """Raised when an exclusion artifact or its source contract is invalid."""


_SOURCE_HASH_CACHE = {}
_SOURCE_HASH_LOCK = threading.Lock()


def clear_source_hash_cache():
    """Clear the process-local source digest cache (primarily for tests)."""
    with _SOURCE_HASH_LOCK:
        _SOURCE_HASH_CACHE.clear()


def _file_identity(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except OSError as exc:
        raise GuardExclusionError("cannot stat source data_path {}: {}".format(path, exc))
    return (
        str(path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _hash_file_uncached(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise GuardExclusionError("cannot hash source data_path {}: {}".format(path, exc))
    return digest.hexdigest()


def source_sha256(path):
    """Hash a source once per unchanged inode/size/mtime/ctime in this process."""
    identity_before = _file_identity(path)
    with _SOURCE_HASH_LOCK:
        cached = _SOURCE_HASH_CACHE.get(identity_before)
    if cached is not None:
        return cached
    digest = _hash_file_uncached(path)
    identity_after = _file_identity(path)
    if identity_after != identity_before:
        raise GuardExclusionError("source data_path changed while SHA256 was computed")
    with _SOURCE_HASH_LOCK:
        stale = [key for key in _SOURCE_HASH_CACHE if key[0] == identity_before[0]]
        for key in stale:
            del _SOURCE_HASH_CACHE[key]
        _SOURCE_HASH_CACHE[identity_before] = digest
    return digest


def _artifact_sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as artifact:
            while True:
                chunk = artifact.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise GuardExclusionError("cannot hash exclusion artifact {}: {}".format(path, exc))
    return digest.hexdigest()


def _verify_artifact_sidecar(path):
    path = Path(path)
    sidecar = path.with_name(path.name + ".sha256")
    try:
        content = sidecar.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise GuardExclusionError("cannot read exclusion SHA256 sidecar {}: {}".format(sidecar, exc))
    lines = content.splitlines()
    if len(lines) != 1 or "  " not in lines[0]:
        raise GuardExclusionError("invalid exclusion SHA256 sidecar format")
    expected, filename = lines[0].split("  ", 1)
    if filename != path.name or len(expected) != 64:
        raise GuardExclusionError("exclusion SHA256 sidecar filename/digest is invalid")
    try:
        int(expected, 16)
    except ValueError:
        raise GuardExclusionError("exclusion SHA256 sidecar digest is not hexadecimal")
    actual = _artifact_sha256(path)
    if actual != expected.lower():
        raise GuardExclusionError(
            "exclusion artifact SHA256 mismatch: expected {}, got {}".format(
                expected.lower(), actual
            )
        )


def _required_int(payload, key):
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GuardExclusionError("{} must be a non-negative integer".format(key))
    return value


def _normalize_id(value):
    if value is None or isinstance(value, (dict, list)):
        raise GuardExclusionError("id must be a non-empty scalar")
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    if not normalized:
        raise GuardExclusionError("id must normalize to a non-empty string")
    return normalized


def _normalize_image(value):
    if not isinstance(value, str) or not value.strip():
        raise GuardExclusionError("image must be a non-empty path string")
    return value.strip()


def _record_identity(record, label):
    if not isinstance(record, dict) or "id" not in record or "image" not in record:
        raise GuardExclusionError("{} must preserve id and image".format(label))
    return _normalize_id(record["id"]), _normalize_image(record["image"])


class GuardExclusion(object):
    """Validated artifact metadata and raw-index identities."""

    def __init__(
        self,
        artifact_path,
        source_path,
        source_digest,
        source_record_count,
        excluded_raw_count,
        excluded_rows,
    ):
        self.artifact_path = str(Path(artifact_path).resolve())
        self.source_path = str(Path(source_path).resolve())
        self.source_digest = source_digest
        self.source_record_count = source_record_count
        self.excluded_raw_count = excluded_raw_count
        self.excluded_rows = excluded_rows
        self.selected_images = frozenset(identity[1] for identity in excluded_rows.values())

    def keep_indices(self, records):
        if len(records) != self.source_record_count:
            raise GuardExclusionError(
                "source_record_count mismatch: artifact {}, loaded {}".format(
                    self.source_record_count, len(records)
                )
            )
        keep = []
        for raw_index, record in enumerate(records):
            source_identity = _record_identity(record, "source raw index {}".format(raw_index))
            expected = self.excluded_rows.get(raw_index)
            if expected is not None:
                if source_identity != expected:
                    raise GuardExclusionError(
                        "raw index {} id/image mismatch: artifact {}, source {}".format(
                            raw_index, expected, source_identity
                        )
                    )
            else:
                if source_identity[1] in self.selected_images:
                    raise GuardExclusionError(
                        "exclusion does not exclude every raw row for selected image {!r}; "
                        "missing raw index {}".format(source_identity[1], raw_index)
                    )
                keep.append(raw_index)
        if len(records) - len(keep) != self.excluded_raw_count:
            raise GuardExclusionError("filtered raw count disagrees with excluded_raw_count")
        return keep


def load_guard_exclusion(exclusion_manifest, data_path):
    """Validate an exclusion artifact, its sidecar, and bound source SHA256."""
    if exclusion_manifest is None:
        return None
    if data_path is None:
        raise GuardExclusionError("data_path is required when exclusion_manifest is set")
    artifact_path = Path(exclusion_manifest)
    _verify_artifact_sidecar(artifact_path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardExclusionError("cannot parse exclusion artifact {}: {}".format(artifact_path, exc))
    if not isinstance(payload, dict):
        raise GuardExclusionError("exclusion artifact must be a JSON object")
    if payload.get("schema_version") != 1:
        raise GuardExclusionError("unsupported exclusion schema_version; expected 1")

    expected_source = payload.get("source_sha256")
    if not isinstance(expected_source, str) or len(expected_source) != 64:
        raise GuardExclusionError("source_sha256 must be a 64-character hexadecimal digest")
    try:
        int(expected_source, 16)
    except ValueError:
        raise GuardExclusionError("source_sha256 must be hexadecimal")
    actual_source = source_sha256(data_path)
    if actual_source != expected_source.lower():
        raise GuardExclusionError(
            "source SHA256 mismatch: artifact {}, data_path {}".format(
                expected_source.lower(), actual_source
            )
        )

    source_record_count = _required_int(payload, "source_record_count")
    excluded_raw_count = _required_int(payload, "excluded_raw_count")
    selected_groups = payload.get("selected_groups")
    if not isinstance(selected_groups, list):
        raise GuardExclusionError("selected_groups must be a JSON array")
    excluded_rows = {}
    for group_index, group in enumerate(selected_groups):
        if not isinstance(group, dict) or not isinstance(group.get("raw_records"), list):
            raise GuardExclusionError(
                "selected_groups[{}].raw_records must be a JSON array".format(group_index)
            )
        if not group["raw_records"]:
            raise GuardExclusionError(
                "selected_groups[{}].raw_records must not be empty".format(group_index)
            )
        group_identity = None
        for row_index, raw_record in enumerate(group["raw_records"]):
            label = "selected_groups[{}].raw_records[{}]".format(group_index, row_index)
            if not isinstance(raw_record, dict):
                raise GuardExclusionError("{} must be a JSON object".format(label))
            raw_index = raw_record.get("raw_index")
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise GuardExclusionError("{}.raw_index must be an integer".format(label))
            if raw_index < 0 or raw_index >= source_record_count:
                raise GuardExclusionError("raw_index {} is out of range".format(raw_index))
            if raw_index in excluded_rows:
                raise GuardExclusionError("duplicate raw_index {} in exclusion".format(raw_index))
            identity = _record_identity(raw_record, label)
            if group_identity is None:
                group_identity = identity
            elif identity != group_identity:
                raise GuardExclusionError(
                    "selected_groups[{}] mixes id/image identities {} and {}".format(
                        group_index, group_identity, identity
                    )
                )
            excluded_rows[raw_index] = identity
    if len(excluded_rows) != excluded_raw_count:
        raise GuardExclusionError(
            "excluded_raw_count {} does not match {} unique raw records".format(
                excluded_raw_count, len(excluded_rows)
            )
        )
    return GuardExclusion(
        artifact_path=artifact_path,
        source_path=data_path,
        source_digest=actual_source,
        source_record_count=source_record_count,
        excluded_raw_count=excluded_raw_count,
        excluded_rows=excluded_rows,
    )


def filter_guard_training_records(records, data_path, exclusion_manifest):
    """Filter raw records in source order, or return the original object if off."""
    exclusion = load_guard_exclusion(exclusion_manifest, data_path)
    if exclusion is None:
        return records
    keep = exclusion.keep_indices(records)
    return [records[index] for index in keep]


def filter_guard_arrow_cache(cache, source_records, exclusion):
    """Validate a full-source Arrow cache, then return its filtered view."""
    if exclusion is None:
        return cache
    keep = exclusion.keep_indices(source_records)
    if len(cache) != exclusion.source_record_count:
        raise GuardExclusionError(
            "Arrow cache length {} is not full source_record_count {}; filtered or stale "
            "cache cannot be trusted".format(len(cache), exclusion.source_record_count)
        )
    column_names = getattr(cache, "column_names", None)
    if not column_names or "id" not in column_names or "image" not in column_names:
        raise GuardExclusionError("Arrow cache must preserve id and image columns")
    if not hasattr(cache, "select"):
        raise GuardExclusionError("Arrow cache does not support deterministic index selection")
    for raw_index, source_record in enumerate(source_records):
        source_identity = _record_identity(
            source_record, "source raw index {}".format(raw_index)
        )
        cache_identity = _record_identity(
            cache[raw_index], "Arrow cache row {}".format(raw_index)
        )
        if cache_identity != source_identity:
            raise GuardExclusionError(
                "Arrow cache order/identity mismatch at raw index {}: source {}, cache {}".format(
                    raw_index, source_identity, cache_identity
                )
            )
    return cache.select(keep)


def load_source_records(data_path):
    """Load JSON/JSONL source records without reordering."""
    path = Path(data_path)
    if path.suffix.lower() == ".jsonl":
        records = []
        try:
            with path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise GuardExclusionError(
                            "invalid source JSONL line {}: {}".format(line_number, exc)
                        )
        except (OSError, UnicodeDecodeError) as exc:
            raise GuardExclusionError("cannot read source JSONL {}: {}".format(path, exc))
    else:
        try:
            with path.open("r", encoding="utf-8") as source:
                payload = json.load(source)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuardExclusionError("cannot read source JSON {}: {}".format(path, exc))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            raise GuardExclusionError("source JSON must be an array or contain records")
    if not all(isinstance(record, dict) for record in records):
        raise GuardExclusionError("every source row must be a JSON object")
    return records
