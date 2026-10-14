from pathlib import Path

from tapeback.vault import obsidian_uri


def test_a_note_opens_in_obsidian_by_its_path():
    """Obsidian finds the vault from the path itself, so the vault's name is not needed."""
    note = Path("/home/me/Notes vault/Voice notes/2026-10-13 standup #2.md")

    assert obsidian_uri(note) == (
        "obsidian://open?path=%2Fhome%2Fme%2FNotes%20vault%2FVoice%20notes"
        "%2F2026-10-13%20standup%20%232.md"
    )
