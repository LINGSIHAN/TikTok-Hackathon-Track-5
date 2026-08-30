import csv
import hashlib
import io
import zipfile
from pathlib import Path, PurePosixPath

import pytest
from PIL import Image

from scripts.download_wildfake_demo import (
    SourceSpec,
    _read_member_bytes,
    _safe_destination,
    _validated_posix_relative,
    download_byte_range,
    download_metadata,
    extract_selected_members,
    select_metadata_rows,
)


class FakeResponse:
    def __init__(self, content, *, status=206, content_range=None):
        self.content = content
        self.status_code = status
        self.headers = {}
        if content_range is not None:
            self.headers["Content-Range"] = content_range
        self.closed = False

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.headers_seen = None

    def get(self, url, **kwargs):
        self.headers_seen = kwargs.get("headers")
        return self.response


def _spec(expected_count=1):
    return SourceSpec(
        name="fixture",
        label=1,
        destination_class="generated",
        metadata_path="dalle3.csv",
        metadata_sha256="0" * 64,
        archive_path="fixture.zip",
        archive_lfs_sha256="1" * 64,
        archive_size=100,
        expected_count=expected_count,
        metadata_prefix="./Diffusion_based/",
        destination_prefix="DALLE/Advanced/DALLE3/",
        required_path_fragment="/Advanced/DALLE3/",
        require_advanced=True,
    )


def _image_bytes(color=(10, 20, 30)):
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(output, format="PNG")
    return output.getvalue()


def _write_metadata(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Image_path", "IsAdvanced", "IsFake"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_metadata_filter_requires_locked_advanced_fake_subset(tmp_path):
    metadata = tmp_path / "dalle3.csv"
    selected_path = (
        "./Diffusion_based/DALLE/Advanced/DALLE3/dalle3/batch/image.png"
    )
    _write_metadata(
        metadata,
        [
            {"Image_path": selected_path, "IsAdvanced": 1, "IsFake": 1},
            {
                "Image_path": "./Diffusion_based/DALLE/Typical/DALLE2/a.png",
                "IsAdvanced": 0,
                "IsFake": 1,
            },
            {
                "Image_path": "./Diffusion_based/DALLE/Advanced/DALLE3/real.png",
                "IsAdvanced": 1,
                "IsFake": 0,
            },
        ],
    )

    selected = select_metadata_rows(metadata, _spec())

    assert [member.archive_name for member in selected] == [
        "DALLE/Advanced/DALLE3/dalle3/batch/image.png"
    ]
    assert selected[0].destination_relative == PurePosixPath(
        "dalle3/batch/image.png"
    )


def test_metadata_count_is_exact(tmp_path):
    metadata = tmp_path / "dalle3.csv"
    _write_metadata(metadata, [])

    with pytest.raises(RuntimeError, match="expected exactly 1"):
        select_metadata_rows(metadata, _spec())


def test_metadata_download_requires_immutable_sha256(tmp_path):
    content = b"Image_path,IsAdvanced,IsFake\n"
    destination = tmp_path / "metadata.csv"

    download_metadata(
        session=FakeSession(FakeResponse(content, status=200)),
        url="https://example.test/metadata.csv",
        destination=destination,
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )
    assert destination.read_bytes() == content

    with pytest.raises(RuntimeError, match="Metadata SHA-256 mismatch"):
        download_metadata(
            session=FakeSession(FakeResponse(b"changed", status=200)),
            url="https://example.test/metadata.csv",
            destination=destination,
            expected_sha256="0" * 64,
        )


def test_range_download_resumes_and_requires_exact_content_range(tmp_path):
    destination = tmp_path / "fixture.range.part"
    destination.write_bytes(b"abc")
    response = FakeResponse(b"defgh", content_range="bytes 13-17/50")
    session = FakeSession(response)

    download_byte_range(
        session=session,
        url="https://example.test/archive.zip",
        destination=destination,
        start=10,
        end=17,
        archive_size=50,
        chunk_size=2,
    )

    assert destination.read_bytes() == b"abcdefgh"
    assert session.headers_seen["Range"] == "bytes=13-17"
    assert response.closed


@pytest.mark.parametrize(
    ("status", "content_range", "message"),
    [
        (200, "bytes 10-17/50", "required HTTP 206"),
        (206, None, "Malformed or missing"),
        (206, "bytes 11-17/50", "Unexpected Content-Range"),
        (206, "nonsense", "Malformed or missing"),
    ],
)
def test_range_download_rejects_malformed_responses(
    tmp_path, status, content_range, message
):
    response = FakeResponse(b"abcdefgh", status=status, content_range=content_range)

    with pytest.raises(RuntimeError, match=message):
        download_byte_range(
            session=FakeSession(response),
            url="https://example.test/archive.zip",
            destination=tmp_path / "fixture.part",
            start=10,
            end=17,
            archive_size=50,
        )


@pytest.mark.parametrize(
    "value",
    ["../escape.png", "/absolute.png", "folder/../../escape.png", "C:/bad.png"],
)
def test_path_traversal_is_rejected(value):
    with pytest.raises(ValueError, match="Unsafe relative path"):
        _validated_posix_relative(value)


def test_destination_stays_under_root(tmp_path):
    root = tmp_path / "images"
    assert _safe_destination(root, PurePosixPath("a/b.png")) == (
        root / "a/b.png"
    ).resolve()


def test_mocked_range_download_and_safe_extraction(tmp_path):
    archive_path = tmp_path / "fixture.zip"
    image_content = _image_bytes()
    archive_name = "DALLE/Advanced/DALLE3/dalle3/batch/image.png"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(archive_name, image_content)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(archive_name)
        range_end = archive.start_dir - 1
    archive_bytes = archive_path.read_bytes()
    range_content = archive_bytes[info.header_offset : range_end + 1]
    range_path = tmp_path / "download.range.part"
    response = FakeResponse(
        range_content,
        content_range=(
            f"bytes {info.header_offset}-{range_end}/{len(archive_bytes)}"
        ),
    )
    download_byte_range(
        session=FakeSession(response),
        url="https://example.test/archive.zip",
        destination=range_path,
        start=info.header_offset,
        end=range_end,
        archive_size=len(archive_bytes),
        chunk_size=7,
    )

    selected = select_metadata_rows_from_fixture(archive_name)
    records = extract_selected_members(
        range_path=range_path,
        range_start=info.header_offset,
        infos=[info],
        selected=selected,
        destination_root=tmp_path / "generated",
    )

    extracted = tmp_path / "generated/dalle3/batch/image.png"
    assert extracted.read_bytes() == image_content
    assert records[0]["path"] == "generated/dalle3/batch/image.png"


def select_metadata_rows_from_fixture(archive_name):
    from scripts.download_wildfake_demo import SelectedMember

    return [
        SelectedMember(
            archive_name=archive_name,
            destination_relative=PurePosixPath("dalle3/batch/image.png"),
            source_path=f"./Diffusion_based/{archive_name}",
        )
    ]


def test_crc_failure_is_rejected(tmp_path):
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("image.png", _image_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("image.png")
        start_dir = archive.start_dir
    content = bytearray(archive_path.read_bytes()[info.header_offset:start_dir])
    content[-1] ^= 0xFF
    range_path = tmp_path / "corrupt.range"
    range_path.write_bytes(content)

    with pytest.raises(RuntimeError, match="deflate|CRC-32|size"):
        _read_member_bytes(range_path, info.header_offset, info)


def test_corrupt_existing_image_is_replaced(tmp_path):
    archive_path = tmp_path / "fixture.zip"
    image_content = _image_bytes((80, 90, 100))
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("image.png", image_content)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("image.png")
        start_dir = archive.start_dir
    range_path = tmp_path / "range"
    range_path.write_bytes(archive_path.read_bytes()[info.header_offset:start_dir])
    destination_root = tmp_path / "generated"
    destination_root.mkdir()
    (destination_root / "image.png").write_bytes(b"not an image")
    from scripts.download_wildfake_demo import SelectedMember

    extract_selected_members(
        range_path=range_path,
        range_start=info.header_offset,
        infos=[info],
        selected=[
            SelectedMember("image.png", PurePosixPath("image.png"), "./image.png")
        ],
        destination_root=destination_root,
    )

    assert (destination_root / "image.png").read_bytes() == image_content
