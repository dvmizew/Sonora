from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.validators import Number

PathArg = Annotated[Path, Parameter(help="Directory containing audio files")]

ThreadsOpt = Annotated[
    int,
    Parameter(
        name=["-t", "--threads"],
        help="Number of parallel threads",
        validator=Number(gte=1),
    ),
]

DryRunOpt = Annotated[
    bool,
    Parameter(negative="", help="Simulate actions without modifying files on disk"),
]

ForceOpt = Annotated[
    bool, Parameter(negative="", help="Force reprocessing even if valid")
]

JsonReportOpt = Annotated[
    Path | None, Parameter(name=["-j", "--json"], help="Save report to JSON file")
]

MusixmatchOpt = Annotated[
    str | None,
    Parameter(name="--musixmatch-token", help="Musixmatch token for enhanced lyrics"),
]

BpmOpt = Annotated[bool, Parameter(name=["--bpm"], help="Calculate audio tempo (BPM)")]

KeyOpt = Annotated[
    bool,
    Parameter(name=["--key"], help="Detect musical key and Camelot wheel tonality"),
]

ReplayGainOpt = Annotated[
    bool,
    Parameter(
        name=["--replaygain"], help="Calculate ReplayGain loudness normalization tags"
    ),
]

LyricsOpt = Annotated[
    bool, Parameter(name=["--lyrics"], help="Fetch synchronized (.lrc) lyrics")
]

ArtOpt = Annotated[
    bool,
    Parameter(name=["--art"], help="Download high-resolution album and artist artwork"),
]

ShazamOpt = Annotated[
    bool,
    Parameter(
        name=["--shazam"],
        help="Enable acoustic recognition via Shazam for untagged tracks",
    ),
]

NoDspOpt = Annotated[
    bool,
    Parameter(
        name=["--no-dsp"],
        negative="",
        help="Disable all heavy audio DSP analysis (BPM, Musical Key, and ReplayGain)",
    ),
]
