"""Fix the diacritic-folding defect in the EP24 codebook coverage auditor.

Bug
---
``_fold`` strips accents by normalising to NFKD and dropping combining marks,
then removes anything that is not ``[a-z0-9 ]``. Two problems follow:

1. **Polish ``ł`` does not decompose.** Unlike ``ż``/``ź``/``ó``, ``ł`` has no
   NFD/NFKD decomposition — it is a distinct letter, not base+diacritic. So it
   survives the combining-mark filter and is then **deleted** by the
   ``[^a-z0-9 ]`` substitution, splitting the word:

   ``Arłukowicz`` -> ``ar ukowicz``   (one token becomes two)

2. This is **language-specific and silent**. It affects any script using
   ``ł``/``đ``/``ı``/``œ`` and similar non-decomposing letters, while Croatian,
   Portuguese, Spanish, German, Swedish, French, Hungarian and Bulgarian are
   unaffected — so a cross-country comparison is distorted in one direction only.

Because ``_tokens`` derives from ``_fold``, the corruption feeds
``entity_fragmentation`` and ``theme_near_duplicates``, inflating fragmentation
counts for exactly the languages it hits.

Fix
---
Treat non-decomposing letters as letters: keep them (mapped to their closest
ASCII base only where that is linguistically safe) instead of deleting them.

The safety question matters. Folding ``ł -> l`` is **not** safe as an identity
rule: Polish ``ł`` is /w/, and ``Łępkowska``/``Lepkowska`` are different names.
So the fix must be conservative:

* keep the letter in the folded key (so the token is not split), and
* never map it onto a *different* letter for identity purposes.

This module demonstrates the corrected behaviour and the measured before/after so
the fix can be reviewed without touching the private data.
"""

from __future__ import annotations

import re
import unicodedata


def fold_broken(value: object) -> str:
    """The current, defective fold — reproduced here for comparison."""
    text = unicodedata.normalize("NFKD", str(value or "").strip().casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fold_fixed(value: object) -> str:
    """Conservative fold that does not destroy non-decomposing letters.

    Differences from the defective version:

    1. Decomposing accents are still stripped (``ż`` -> ``z``, ``ó`` -> ``o``).
    2. Letters that do NOT decompose are **kept as themselves** rather than being
       deleted, so words are never split. This preserves the distinction between
       ``ł`` and ``l``.
    3. Punctuation that is genuinely punctuation is still removed.
    """
    text = unicodedata.normalize("NFKD", str(value or "").strip().casefold())
    # Drop only combining marks (true accents), keep every base letter.
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove punctuation but NOT letters of any script.
    text = "".join(c if (c.isalnum() or c.isspace()) else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def non_decomposing_letters() -> list[str]:
    """Non-ASCII letters that survive NFKD unchanged (the bug's blast radius).

    Original scope was the Latin Extended-A/B blocks (``U+0100``–``U+024F``).
    That was too narrow in a way that hid the worst instance of the defect: the
    scan must cover **every script**, because the class is "letters NFKD cannot
    decompose", and the largest such class in this project's data is Cyrillic.

    The Bulgarian pass (#91) found that the old ``[^a-z0-9 ]`` filter deleted
    *all* 80+ non-decomposing Cyrillic capital letters, so BG labels folded to
    the empty string. A scan restricted to a Latin block cannot report that, and
    therefore cannot warn the fixer that the Cyrillic script is affected at all.
    """
    out = []
    for cp in range(0x80, 0x30000):
        ch = chr(cp)
        if not unicodedata.category(ch).startswith("L"):
            continue
        if unicodedata.normalize("NFKD", ch) != ch:
            continue
        if ch.isascii():
            continue
        out.append(ch)
    return out


def non_decomposing_scripts() -> dict[str, int]:
    """How many non-decomposing letters each script contributes.

    A per-script count, so a reviewer can see at a glance which writing systems
    a folding change would affect. ``Cyrillic`` and ``Latin`` dominate the EP24
    data, but the number is derived rather than asserted.

    Unicode names are all-uppercase ("CYRILLIC CAPITAL LETTER GHE"), so the
    leading token is title-cased to make the report readable.
    """
    counts: dict[str, int] = {}
    for ch in non_decomposing_letters():
        try:
            name = unicodedata.name(ch)
        except ValueError:  # pragma: no cover - unnamed code points
            name = ""
        script = name.split(" ")[0].capitalize() if name else "Unknown"
        counts[script] = counts.get(script, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


if __name__ == "__main__":  # pragma: no cover - manual demonstration
    for label in ("Arłukowicz", "Łępkowska", "Bożena Przyłuska", "Kraków", "Żukowska"):
        b, f = fold_broken(label), fold_fixed(label)
        flag = "SPLIT" if b.count(" ") != f.count(" ") else "ok"
        print(f"{label:22} broken={b!r:28} fixed={f!r:28} {flag}")
