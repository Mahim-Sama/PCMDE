"""
Subtype resolver (v2).

Given a caption and/or filename, identify the subtype of the primary object
in the image. Configuration is taken directly from the rules JSON's
"subtypes" and "fallback_subtype" blocks; no separate knowledge file.

Expected structure in the rules JSON:

    "subtypes": {
      "<subtype_id>": {
        "display_name":            "<string>",
        "filename_prefix_patterns": ["<prefix>", ...],
        "caption_regex_patterns":   ["<regex>", ...]
      },
      ...
    },
    "fallback_subtype": {
      "subtype_id":   "<string>",
      "display_name": "<string>"
    }

Resolution priority (per image):
    1. Caption regex match (if a caption is supplied). When patterns of several
       subtypes match, the subtype mentioned most often wins; ties go to the
       subtype mentioned first. Long descriptive captions often name other
       objects in passing ("a Dalmatian ... trotting alongside horse-drawn
       carriages"), so neither the longest match nor declaration order is a
       reliable signal of the primary object.
    2. Filename prefix match
    3. Fallback subtype
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SubtypeResolution:
    """
    Outcome of subtype resolution for one image.

    Attributes:
        subtype_id:       Canonical id used for rule matching.
        display_name:     Human-readable name.
        is_fallback:      True when no pattern matched and the fallback was used.
        source:           One of "caption", "filename", "fallback".
        matched_pattern:  The regex or prefix string that matched, for diagnostics.
    """
    subtype_id: str
    display_name: str
    is_fallback: bool
    source: str
    matched_pattern: Optional[str] = None


class SubtypeResolver:
    """
    Resolves image subtypes using patterns declared in the rules JSON itself.
    """

    def __init__(self,
                 subtypes: Dict[str, Dict[str, Any]],
                 fallback_subtype: Dict[str, str]):
        self._caption_patterns: List[Dict[str, Any]] = []
        self._filename_prefix_patterns: List[Dict[str, Any]] = []

        for subtype_id, info in (subtypes or {}).items():
            display_name = info.get("display_name", subtype_id)

            # Compile caption regex patterns
            for regex_str in info.get("caption_regex_patterns", []) or []:
                try:
                    compiled = re.compile(regex_str, re.IGNORECASE)
                except re.error:
                    continue
                self._caption_patterns.append({
                    "regex": compiled,
                    "source_str": regex_str,
                    "subtype_id": subtype_id,
                    "display_name": display_name,
                })

            # Collect filename prefix patterns
            for prefix in info.get("filename_prefix_patterns", []) or []:
                self._filename_prefix_patterns.append({
                    "prefix": prefix,
                    "subtype_id": subtype_id,
                    "display_name": display_name,
                })

        self._fallback_id = (fallback_subtype or {}).get("subtype_id", "generic")
        self._fallback_name = (fallback_subtype or {}).get("display_name", "Generic")

    def resolve(self,
                caption: Optional[str] = None,
                filename: Optional[str] = None) -> SubtypeResolution:
        """
        Resolve the subtype for a given image. Priority: caption > filename > fallback.
        """
        # 1. Caption regex match: most mentions wins, ties -> earliest mention
        if caption:
            # subtype_id -> [mention_count, first_position, entry, first_match_text]
            tally: Dict[str, List[Any]] = {}
            for entry in self._caption_patterns:
                for m in entry["regex"].finditer(caption):
                    stats = tally.get(entry["subtype_id"])
                    if stats is None:
                        tally[entry["subtype_id"]] = [1, m.start(), entry, m.group(0)]
                        continue
                    stats[0] += 1
                    if m.start() < stats[1]:
                        stats[1], stats[2], stats[3] = m.start(), entry, m.group(0)
            if tally:
                count, _, best_entry, match_text = min(
                    tally.values(), key=lambda s: (-s[0], s[1])
                )
                return SubtypeResolution(
                    subtype_id=best_entry["subtype_id"],
                    display_name=best_entry["display_name"],
                    is_fallback=False,
                    source="caption",
                    matched_pattern=(
                        f"{best_entry['source_str']} "
                        f"(matched: {match_text!r}, mentions: {count})"
                    ),
                )

        # 2. Filename prefix match
        if filename:
            name = Path(filename).name.lower()
            for entry in self._filename_prefix_patterns:
                prefix = entry["prefix"].lower()
                if (name.startswith(prefix + "_")
                        or name.startswith(prefix + ".")
                        or name == prefix):
                    return SubtypeResolution(
                        subtype_id=entry["subtype_id"],
                        display_name=entry["display_name"],
                        is_fallback=False,
                        source="filename",
                        matched_pattern=f"prefix:{prefix}",
                    )

        # 3. Fallback
        return SubtypeResolution(
            subtype_id=self._fallback_id,
            display_name=self._fallback_name,
            is_fallback=True,
            source="fallback",
        )