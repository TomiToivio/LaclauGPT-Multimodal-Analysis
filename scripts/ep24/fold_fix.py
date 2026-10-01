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
    """Latin-script letters that survive NFKD unchanged (the bug's blast radius)."""
    out = []
    for cp in range(0x100, 0x250):
        ch = chr(cp)
        if (
            unicodedata.category(ch).startswith("L")
            and unicodedata.normalize("NFKD", ch) == ch
            and ch.isascii() is False
        ):
            out.append(ch)
    return out


if __name__ == "__main__":  # pragma: no cover - manual demonstration
    for label in ("Arłukowicz", "Łępkowska", "Bożena Przyłuska", "Kraków", "Żukowska"):
        b, f = fold_broken(label), fold_fixed(label)
        flag = "SPLIT" if b.count(" ") != f.count(" ") else "ok"
        print(f"{label:22} broken={b!r:28} fixed={f!r:28} {flag}")
