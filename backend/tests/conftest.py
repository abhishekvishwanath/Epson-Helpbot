import os
import tempfile

os.environ.setdefault("ENVIRONMENT", "test")
# Never touch the real session/registry database from tests.
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp(prefix='psa-test-')}/test.db"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402


@pytest.fixture(scope="session")
def manual_path():
    path = get_settings().default_document_path
    if not path.is_file():
        pytest.skip(f"Manual PDF not present at {path}")
    return path


@pytest.fixture(scope="session")
def parsed_manual(manual_path, tmp_path_factory):
    """Parse the real manual once per test session (cached processed output reused if current)."""
    from app.ingestion.document_builder import file_sha256, load_parsed, parse_document

    s = get_settings()
    cached = s.processed_dir / s.default_document_id
    if (cached / "structure.json").is_file():
        doc = load_parsed(cached)
        if doc.source_hash == file_sha256(manual_path):
            return doc
    out = tmp_path_factory.mktemp("parsed")
    return parse_document(
        manual_path,
        document_id=s.default_document_id,
        document_name=s.default_document_name,
        manufacturer=s.default_manufacturer,
        product_model=s.default_product_model,
        figures_dir=out / "figures",
        ocr_enabled=False,
    )
