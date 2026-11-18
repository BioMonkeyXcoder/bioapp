"""Package the scientific wheels against the system-installed Visual C++ runtime."""

import re
from pathlib import Path

import pefile


# The locked NumPy and pandas wheels use a private basename for this runtime.
PRIVATE_MSVCP = re.compile(rb"msvcp140-[0-9a-f]{32}\.dll", re.IGNORECASE)
RUNTIME_FILE = re.compile(
    r"(?:msvcp140(?:_\d+|_atomic_wait|_codecvt_ids)?|vcruntime140(?:_\d+(?:_clr0400)?)?"
    r"|ucrtbase|api-ms-win-[\w-]+)"
    r"(?:-[0-9a-f]{32})?\.dll", re.IGNORECASE
)


def system_runtime_binary(data):
    """Return the input unless an actual DLL import needs the system basename."""
    if not PRIVATE_MSVCP.search(data):
        return data
    with pefile.PE(data=data, fast_load=True) as pe:
        security = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"]]
        if security.Size:
            raise ValueError("A signed binary requests a private runtime; review the new wheel.")
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
        ])
        updated = bytearray(data)
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            if PRIVATE_MSVCP.fullmatch(entry.dll):
                start = pe.get_offset_from_rva(entry.struct.Name)
                if data[start:start + len(entry.dll) + 1] != entry.dll + b"\0":
                    raise ValueError("Unexpected import descriptor for the VC++ runtime.")
                updated[start:start + len(entry.dll)] = b"msvcp140.dll".ljust(len(entry.dll), b"\0")
        if PRIVATE_MSVCP.search(updated):
            raise ValueError("An unhandled reference to a private runtime remains.")
    with pefile.PE(data=bytes(updated), fast_load=True) as rebound:
        rebound.OPTIONAL_HEADER.CheckSum = rebound.generate_checksum()
        return rebound.write()


def prepare_collection(entries, build_dir):
    """Stage only changed extensions; leave installed wheels intact."""
    result = []
    for dest, src, kind in entries:
        name = Path(dest)
        if RUNTIME_FILE.fullmatch(name.name):
            continue
        if name.suffix.lower() in {".dll", ".pyd"}:
            original = Path(src).read_bytes()
            updated = system_runtime_binary(original)
            if updated is not original:
                staged = Path(build_dir) / "system-runtime" / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(updated)
                src = str(staged)
        result.append((dest, src, kind))
    return result
