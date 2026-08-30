"""Download and verify the exact WildFake demonstration subset.

Only the two contiguous ZIP member spans needed by the benchmark are fetched.
The source archives are never downloaded in full and extracted image bytes are
preserved exactly as encoded upstream.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import io
import json
import re
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode
from zipfile import ZIP_DEFLATED, ZipInfo

import requests
from PIL import Image
from remotezip import RemoteFetcher, RemoteZip, RemoteZipError


REPOSITORY = "hy2628982280/WildFake"
REPOSITORY_URL = f"https://www.modelscope.cn/datasets/{REPOSITORY}"
REVISION = "18f53ff36ad9da60644039f0452b0e7b3907af6f"
USER_AGENT = "RealityCheck-WildFake-Demo/1.0"
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_HEADER_SIGNATURE = 0x04034B50
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    label: int
    destination_class: str
    metadata_path: str
    metadata_sha256: str
    archive_path: str
    archive_lfs_sha256: str
    archive_size: int
    expected_count: int
    metadata_prefix: str
    destination_prefix: str
    required_path_fragment: str
    require_advanced: bool


SOURCES = (
    SourceSpec(
        name="COCO val2017",
        label=0,
        destination_class="real",
        metadata_path="label_csv_files/real_coco.csv",
        metadata_sha256=(
            "b4903f9302361a838348c8d55a11f435aacceecd9926535488ec4de4f172e179"
        ),
        archive_path="Images/Real/coco.zip",
        archive_lfs_sha256=(
            "0b4dda0968e5f0d3cb60434c24204fcdac1cc0b40018093f15307edd545905b3"
        ),
        archive_size=2_353_803_219,
        expected_count=4_998,
        metadata_prefix="./Real/",
        destination_prefix="coco/",
        required_path_fragment="/val2017/",
        require_advanced=False,
    ),
    SourceSpec(
        name="Advanced DALL-E 3",
        label=1,
        destination_class="generated",
        metadata_path="label_csv_files/dalle3.csv",
        metadata_sha256=(
            "b490818255b2c1e3c6d07d271d29e0b9488de9b8c1338e0cf0286ecc6ff574ed"
        ),
        archive_path="Images/Diffusion_based/DALLE.zip",
        archive_lfs_sha256=(
            "5e4ebc56daa06ebeec99711b9cc204571558d3e17366f2df992a8cfd4f251d4c"
        ),
        archive_size=25_587_709_291,
        expected_count=8_843,
        metadata_prefix="./Diffusion_based/",
        destination_prefix="DALLE/Advanced/DALLE3/",
        required_path_fragment="/Advanced/DALLE3/",
        require_advanced=True,
    ),
)


@dataclass(frozen=True)
class SelectedMember:
    archive_name: str
    destination_relative: PurePosixPath
    source_path: str


@dataclass(frozen=True)
class ArchiveRange:
    start: int
    end: int
    members: tuple[ZipInfo, ...]

    @property
    def size(self) -> int:
        return self.end - self.start + 1


class KnownSizeRemoteFetcher(RemoteFetcher):
    """RemoteZip fetcher for endpoints whose HEAD omits Content-Length."""

    def __init__(
        self,
        url: str,
        session: requests.Session | None = None,
        *,
        known_size: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(url, session, **kwargs)
        self.known_size = known_size

    def get_file_size(self) -> int:
        return self.known_size

    def _request(self, kwargs: dict[str, Any]) -> tuple[Any, str]:
        raw, content_range = super()._request(kwargs)
        match = CONTENT_RANGE.fullmatch(content_range)
        if match is None or int(match.group(3)) != self.known_size:
            raw.close()
            if hasattr(raw, "release_conn"):
                raw.release_conn()
            raise RuntimeError(
                f"Remote ZIP returned an invalid Content-Range: {content_range!r}"
            )
        return raw, content_range


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def modelscope_url(file_path: str) -> str:
    query = urlencode({"Revision": REVISION, "FilePath": file_path})
    return f"https://www.modelscope.cn/api/v1/datasets/{REPOSITORY}/repo?{query}"


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
    temporary.replace(path)


def download_metadata(
    *,
    session: requests.Session,
    url: str,
    destination: Path,
    expected_sha256: str,
) -> None:
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return
    response = session.get(url, timeout=(30, 120))
    response.raise_for_status()
    content = response.content
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Metadata SHA-256 mismatch for {destination.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    _atomic_write_bytes(destination, content)


def _parse_binary_flag(value: str, column: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Metadata row {row_number} has invalid {column}={value!r}"
        ) from error
    if parsed not in (0, 1):
        raise RuntimeError(
            f"Metadata row {row_number} has invalid {column}={value!r}"
        )
    return parsed


def _validated_posix_relative(value: str) -> PurePosixPath:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    if any(part in ("", ".", "..") for part in candidate.parts):
        raise ValueError(f"Unsafe relative path: {value!r}")
    if "\\" in value or candidate.parts[0].endswith(":"):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return candidate


def select_metadata_rows(metadata_path: Path, spec: SourceSpec) -> list[SelectedMember]:
    selected: list[SelectedMember] = []
    seen: set[str] = set()
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Image_path", "IsAdvanced", "IsFake"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(
                f"{metadata_path.name} is missing required columns: "
                f"{', '.join(sorted(required))}"
            )
        for row_number, row in enumerate(reader, start=2):
            source_path = str(row["Image_path"])
            is_advanced = _parse_binary_flag(
                row["IsAdvanced"], "IsAdvanced", row_number
            )
            is_fake = _parse_binary_flag(row["IsFake"], "IsFake", row_number)
            include = (
                spec.required_path_fragment in source_path
                and is_fake == spec.label
                and (not spec.require_advanced or is_advanced == 1)
            )
            if not include:
                continue
            if not source_path.startswith(spec.metadata_prefix):
                raise RuntimeError(
                    f"Unexpected source path in {metadata_path.name}: {source_path}"
                )
            archive_name = source_path[len(spec.metadata_prefix) :]
            if not archive_name.startswith(spec.destination_prefix):
                raise RuntimeError(f"Unexpected archive member path: {archive_name}")
            destination_value = archive_name[len(spec.destination_prefix) :]
            destination_relative = _validated_posix_relative(destination_value)
            _validated_posix_relative(archive_name)
            if archive_name in seen:
                raise RuntimeError(f"Duplicate metadata image path: {archive_name}")
            seen.add(archive_name)
            selected.append(
                SelectedMember(
                    archive_name=archive_name,
                    destination_relative=destination_relative,
                    source_path=source_path,
                )
            )
    selected.sort(key=lambda member: member.archive_name)
    if len(selected) != spec.expected_count:
        raise RuntimeError(
            f"{spec.name} metadata selected {len(selected):,} images; "
            f"expected exactly {spec.expected_count:,}"
        )
    return selected


def inspect_archive(
    *,
    url: str,
    selected: Sequence[SelectedMember],
    expected_archive_size: int,
    session: requests.Session,
) -> ArchiveRange:
    requested_names = {member.archive_name for member in selected}
    with RemoteZip(
        url,
        session=session,
        support_suffix_range=False,
        fetcher=KnownSizeRemoteFetcher,
        known_size=expected_archive_size,
        headers={"User-Agent": USER_AGENT},
    ) as archive:
        infos = list(archive.infolist())
        archive_size = int(archive.size())
        start_dir = int(archive.start_dir)
    if archive_size != expected_archive_size:
        raise RuntimeError(
            f"Archive size mismatch: expected {expected_archive_size:,}, "
            f"got {archive_size:,}"
        )

    info_by_name = {info.filename: info for info in infos}
    missing = requested_names - info_by_name.keys()
    if missing:
        example = sorted(missing)[0]
        raise RuntimeError(
            f"Archive is missing {len(missing):,} selected member(s), e.g. {example}"
        )
    target_infos = sorted(
        (info_by_name[name] for name in requested_names),
        key=lambda info: info.header_offset,
    )
    for info in target_infos:
        if info.is_dir():
            raise RuntimeError(f"Selected archive member is a directory: {info.filename}")
        if info.compress_type != ZIP_DEFLATED:
            raise RuntimeError(
                f"Unsupported compression method for {info.filename}: {info.compress_type}"
            )
        if info.flag_bits != 0:
            raise RuntimeError(
                f"Unexpected ZIP flags for {info.filename}: {info.flag_bits:#x}"
            )

    ordered_infos = sorted(infos, key=lambda info: info.header_offset)
    first_index = ordered_infos.index(target_infos[0])
    last_index = ordered_infos.index(target_infos[-1])
    interleaved = [
        info.filename
        for info in ordered_infos[first_index : last_index + 1]
        if not info.is_dir() and info.filename not in requested_names
    ]
    if interleaved:
        raise RuntimeError(
            "Selected ZIP members are not a contiguous range; first unexpected "
            f"member: {interleaved[0]}"
        )
    next_offset = (
        ordered_infos[last_index + 1].header_offset
        if last_index + 1 < len(ordered_infos)
        else start_dir
    )
    return ArchiveRange(
        start=target_infos[0].header_offset,
        end=next_offset - 1,
        members=tuple(target_infos),
    )


def _validate_content_range(
    header: str | None,
    *,
    requested_start: int,
    requested_end: int,
    expected_total: int,
) -> None:
    match = CONTENT_RANGE.fullmatch(header or "")
    if match is None:
        raise RuntimeError(f"Malformed or missing Content-Range response: {header!r}")
    actual_start, actual_end, actual_total = (int(value) for value in match.groups())
    if (actual_start, actual_end, actual_total) != (
        requested_start,
        requested_end,
        expected_total,
    ):
        raise RuntimeError(
            "Unexpected Content-Range response: "
            f"{header!r}; expected bytes {requested_start}-{requested_end}/"
            f"{expected_total}"
        )


def download_byte_range(
    *,
    session: requests.Session,
    url: str,
    destination: Path,
    start: int,
    end: int,
    archive_size: int,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    expected_size = end - start + 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0
    if existing_size > expected_size:
        raise RuntimeError(
            f"Partial range file is too large: {existing_size:,} > {expected_size:,}"
        )
    if existing_size == expected_size:
        return

    requested_start = start + existing_size
    response = session.get(
        url,
        headers={
            "Range": f"bytes={requested_start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": USER_AGENT,
        },
        stream=True,
        timeout=(30, 120),
    )
    if response.status_code != 206:
        response.close()
        raise RuntimeError(
            f"Range request returned HTTP {response.status_code}; required HTTP 206"
        )
    _validate_content_range(
        response.headers.get("Content-Range"),
        requested_start=requested_start,
        requested_end=end,
        expected_total=archive_size,
    )

    started = time.monotonic()
    next_progress = existing_size + 256 * 1024 * 1024
    try:
        with destination.open("ab") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                current_size = handle.tell()
                if current_size >= next_progress:
                    elapsed = max(time.monotonic() - started, 0.001)
                    downloaded = current_size - existing_size
                    speed = downloaded / elapsed / (1024 * 1024)
                    print(
                        f"  downloaded {current_size / (1024 ** 3):.2f}/"
                        f"{expected_size / (1024 ** 3):.2f} GiB ({speed:.1f} MiB/s)",
                        flush=True,
                    )
                    next_progress += 256 * 1024 * 1024
            handle.flush()
    finally:
        response.close()
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Incomplete byte range: expected {expected_size:,} bytes, got "
            f"{actual_size:,}; rerun to resume"
        )


def _safe_destination(root: Path, relative: PurePosixPath) -> Path:
    validated = _validated_posix_relative(relative.as_posix())
    root = root.resolve()
    destination = root.joinpath(*validated.parts).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Destination escapes output root: {relative}") from error
    return destination


def validate_image_bytes(content: bytes, *, name: str) -> None:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise OSError("image has invalid dimensions")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Image could not be decoded: {name}") from error


def verify_extracted_file(path: Path, info: ZipInfo) -> tuple[bool, str | None]:
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False, None
    try:
        content = path.read_bytes()
        if binascii.crc32(content) & 0xFFFFFFFF != info.CRC:
            return False, None
        validate_image_bytes(content, name=str(path))
    except (OSError, RuntimeError):
        return False, None
    return True, hashlib.sha256(content).hexdigest()


def _read_member_bytes(range_path: Path, range_start: int, info: ZipInfo) -> bytes:
    relative_offset = info.header_offset - range_start
    if relative_offset < 0:
        raise RuntimeError(f"Member starts before downloaded range: {info.filename}")
    with range_path.open("rb") as handle:
        handle.seek(relative_offset)
        raw_header = handle.read(LOCAL_HEADER.size)
        if len(raw_header) != LOCAL_HEADER.size:
            raise RuntimeError(f"Truncated local ZIP header: {info.filename}")
        (
            signature,
            _version,
            flags,
            compression,
            _modified_time,
            _modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_size,
            extra_size,
        ) = LOCAL_HEADER.unpack(raw_header)
        if signature != LOCAL_HEADER_SIGNATURE:
            raise RuntimeError(f"Invalid local ZIP header: {info.filename}")
        if flags != info.flag_bits or flags & 0x1:
            raise RuntimeError(f"Unsafe or inconsistent ZIP flags: {info.filename}")
        if compression != ZIP_DEFLATED or compression != info.compress_type:
            raise RuntimeError(f"Unsupported ZIP compression: {info.filename}")
        if (crc32, compressed_size, uncompressed_size) != (
            info.CRC,
            info.compress_size,
            info.file_size,
        ):
            raise RuntimeError(f"Local and central ZIP metadata differ: {info.filename}")
        raw_filename = handle.read(filename_size)
        encoding = "utf-8" if flags & 0x800 else "cp437"
        if raw_filename.decode(encoding) != info.filename:
            raise RuntimeError(f"Local ZIP filename mismatch: {info.filename}")
        handle.seek(extra_size, 1)
        compressed = handle.read(info.compress_size)
        if len(compressed) != info.compress_size:
            raise RuntimeError(f"Truncated compressed member: {info.filename}")

    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    content = inflater.decompress(compressed) + inflater.flush()
    if not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise RuntimeError(f"Invalid deflate stream: {info.filename}")
    if len(content) != info.file_size:
        raise RuntimeError(f"Uncompressed size mismatch: {info.filename}")
    actual_crc = binascii.crc32(content) & 0xFFFFFFFF
    if actual_crc != info.CRC:
        raise RuntimeError(
            f"CRC-32 mismatch for {info.filename}: expected {info.CRC:08x}, "
            f"got {actual_crc:08x}"
        )
    validate_image_bytes(content, name=info.filename)
    return content


def extract_selected_members(
    *,
    range_path: Path,
    range_start: int,
    infos: Sequence[ZipInfo],
    selected: Sequence[SelectedMember],
    destination_root: Path,
) -> list[dict[str, Any]]:
    selected_by_name = {member.archive_name: member for member in selected}
    records: list[dict[str, Any]] = []
    for index, info in enumerate(infos, start=1):
        selected_member = selected_by_name[info.filename]
        destination = _safe_destination(
            destination_root, selected_member.destination_relative
        )
        valid, digest = verify_extracted_file(destination, info)
        if not valid:
            content = _read_member_bytes(range_path, range_start, info)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with temporary.open("wb") as handle:
                handle.write(content)
                handle.flush()
            temporary.replace(destination)
            valid, digest = verify_extracted_file(destination, info)
            if not valid:
                raise RuntimeError(f"Extracted image failed verification: {info.filename}")
        assert digest is not None
        records.append(
            {
                "path": (
                    PurePosixPath(destination_root.name)
                    / selected_member.destination_relative
                ).as_posix(),
                "source_path": selected_member.source_path,
                "sha256": digest,
                "size": info.file_size,
                "crc32": f"{info.CRC:08x}",
            }
        )
        if index % 500 == 0 or index == len(infos):
            print(f"  verified {index:,}/{len(infos):,} {destination_root.name} images")
    return records


def aggregate_dataset_digest(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    ordered = sorted(records, key=lambda record: str(record["path"]))
    for record in ordered:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_download(root: Path) -> dict[str, Any]:
    root = root.resolve()
    metadata_root = root / "_metadata"
    temporary_root = root / ".ranges"
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    all_records: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    try:
        for spec in SOURCES:
            print(f"Preparing {spec.name} metadata...", flush=True)
            metadata_path = metadata_root / Path(spec.metadata_path).name
            download_metadata(
                session=session,
                url=modelscope_url(spec.metadata_path),
                destination=metadata_path,
                expected_sha256=spec.metadata_sha256,
            )
            selected = select_metadata_rows(metadata_path, spec)
            archive_url = modelscope_url(spec.archive_path)
            print(f"Inspecting {spec.name} remote ZIP directory...", flush=True)
            archive_range = inspect_archive(
                url=archive_url,
                selected=selected,
                expected_archive_size=spec.archive_size,
                session=session,
            )
            range_path = temporary_root / f"{spec.destination_class}.range.part"
            print(
                f"Downloading {spec.name} range "
                f"({archive_range.size / (1024 ** 3):.2f} GiB)...",
                flush=True,
            )
            download_byte_range(
                session=session,
                url=archive_url,
                destination=range_path,
                start=archive_range.start,
                end=archive_range.end,
                archive_size=spec.archive_size,
            )
            print(f"Extracting and verifying {spec.name}...", flush=True)
            records = extract_selected_members(
                range_path=range_path,
                range_start=archive_range.start,
                infos=archive_range.members,
                selected=selected,
                destination_root=root / spec.destination_class,
            )
            if len(records) != spec.expected_count:
                raise RuntimeError(
                    f"Verified {len(records):,} {spec.name} images; expected "
                    f"{spec.expected_count:,}"
                )
            range_path.unlink()
            all_records.extend(records)
            source_summaries.append(
                {
                    "name": spec.name,
                    "label": spec.label,
                    "class_directory": spec.destination_class,
                    "count": len(records),
                    "metadata_path": spec.metadata_path,
                    "metadata_sha256": spec.metadata_sha256,
                    "archive_path": spec.archive_path,
                    "archive_lfs_sha256": spec.archive_lfs_sha256,
                    "archive_size": spec.archive_size,
                    "downloaded_range": {
                        "start": archive_range.start,
                        "end": archive_range.end,
                        "size": archive_range.size,
                    },
                }
            )
    finally:
        session.close()

    manifest = {
        "schema_version": 1,
        "repository": REPOSITORY_URL,
        "revision": REVISION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_digest_algorithm": "sha256(sorted(path + NUL + file_sha256 + LF))",
        "dataset_digest": aggregate_dataset_digest(all_records),
        "total_count": len(all_records),
        "class_counts": {
            spec.destination_class: spec.expected_count for spec in SOURCES
        },
        "sources": source_summaries,
        "files": all_records,
    }
    if manifest["total_count"] != sum(spec.expected_count for spec in SOURCES):
        raise RuntimeError("Final WildFake count does not match the locked subset")
    _write_json(root / "download_manifest.json", manifest)
    try:
        temporary_root.rmdir()
    except OSError:
        pass
    print(
        f"WildFake demonstration subset ready: {manifest['total_count']:,} verified "
        f"images; digest {manifest['dataset_digest']}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Download the exact immutable WildFake demonstration subset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repository_root / "data/external/wildfake_demo",
        help="Destination dataset root (default: data/external/wildfake_demo)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_download(args.root)
    except (
        OSError,
        RemoteZipError,
        RuntimeError,
        ValueError,
        requests.RequestException,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
