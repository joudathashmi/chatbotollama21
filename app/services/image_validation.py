"""
Real image-content validation for uploads.

The client-declared `Content-Type` header (checked against
`ALLOWED_MIME_TYPES` in the router) is attacker-controlled — nothing
stops a client from labelling an executable, script, or oversized
polyglot file `image/png`. This module verifies the ACTUAL bytes are a
real image matching the declared format before anything is handed to
Google Vision / OpenAI.

Two tiers, by necessity:
  - JPEG / PNG / WebP: full decode via Pillow's `Image.verify()`, which
    parses the real pixel-format structure — not just a header sniff.
  - HEIC / HEIF: signature-only (ISO-BMFF `ftyp` box + a known brand).
    Base Pillow cannot decode HEIC without the optional `pillow-heif`
    plugin (which itself needs the system `libheif` library — not
    something to silently require here). This is a real but narrower
    check than the other three formats; documented as such rather than
    pretended to be a full decode.
"""

from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

# Content-Type -> the Pillow format string(s) it must actually decode to.
_PILLOW_VERIFIABLE: dict[str, set[str]] = {
    "image/jpeg": {"JPEG"},
    "image/png": {"PNG"},
    "image/webp": {"WEBP"},
}

# ISO-BMFF brand identifiers found in a HEIC/HEIF file's leading 'ftyp' box.
_HEIC_BRANDS: set[bytes] = {
    b"heic", b"heix", b"hevc", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1",
}


def sniff_and_validate(image_bytes: bytes, declared_content_type: str) -> tuple[bool, str | None]:
    """Verify `image_bytes` is a real image matching `declared_content_type`.

    Returns (is_valid, reason) — `reason` is a client-safe explanation
    string when invalid, `None` when valid.
    """
    content_type = (declared_content_type or "").strip().lower()

    if content_type in _PILLOW_VERIFIABLE:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            fmt = img.format  # read before verify() invalidates the handle
            img.verify()
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
            return False, f"File content is not a valid, decodable image ({e})."

        expected = _PILLOW_VERIFIABLE[content_type]
        if fmt not in expected:
            return False, (
                f"File content is actually a {fmt or 'unknown'} image, but "
                f"Content-Type declared {content_type}."
            )
        return True, None

    if content_type in ("image/heic", "image/heif"):
        return _sniff_heic(image_bytes)

    # ALLOWED_MIME_TYPES in the router already gates this before we're
    # called — fail closed if an unrecognized type ever reaches here.
    return False, f"Unrecognized content type for validation: {content_type!r}"


def _sniff_heic(image_bytes: bytes) -> tuple[bool, str | None]:
    if len(image_bytes) < 12:
        return False, "File is too short to be a valid HEIC/HEIF image."
    if image_bytes[4:8] != b"ftyp":
        return False, "File does not have a valid HEIC/HEIF (ISO-BMFF) header."
    brand = image_bytes[8:12]
    if brand not in _HEIC_BRANDS:
        return False, f"Unrecognized HEIC/HEIF brand marker: {brand!r}."
    return True, None
