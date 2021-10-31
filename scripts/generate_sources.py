"""
Generates and saves file containing list of sources for the project.
"""


from pathlib import Path
from argparse import ArgumentParser, Namespace
import os
from collections.abc import Mapping, Iterable
from typing import Any, Optional


ArgsSpec = Mapping[Iterable[str], Mapping[str, Any]]


def validate_args(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise RuntimeError(f"source path `{source}` is not a directory")

    if not destination.is_file():
        raise RuntimeError(f"destination path `{destination}` is not a file")

    if not source.exists():
        raise RuntimeError(f"source path `{source}` does not exist")


def setup_args(args_spec: ArgsSpec, arg_parser: ArgumentParser) -> Namespace:
    for spec in args_spec.items():
        name_or_flags, options = spec
        arg_parser.add_argument(*name_or_flags, **options)

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    here = Path(__file__)
    sources_path = here.parent / "../docs/sources/"
    target_path = sources_path / "source_list.txt"

    args_spec: ArgsSpec = {
        ("-s", "--source"): {"default": sources_path, "type": Path},
        ("-d", "--destination"): {"default": target_path, "type": Path},
        ("--previous_source",): {"nargs": "?", "default": None, "type": Optional[Path]},
    }

    arg_parser = ArgumentParser()

    args = setup_args(args_spec, arg_parser)

    source: Path = args.source
    destination: Path = args.destination
    to_exclude = [destination] + (args.previous_source is not None) * [
        args.previous_source
    ]

    # validate_args(source, destination)

    if not destination.parent.exists():
        os.makedirs(destination, exist_ok=True)

    if destination.exists():
        destination_did_exist = True
        with destination.open("r") as f:
            destination_backup = f.read()
    else:
        destination_backup = ""
        destination_did_exist = False

    try:
        with destination.open("w") as f:
            source_list = [
                path.parts[-1]
                for path in source.iterdir()
                if path.is_file() and path not in to_exclude
            ]
            f.write("\n".join(map(str, source_list)))
    except Exception as e:
        if not destination_did_exist:
            raise e

        with destination.open("w") as f:
            f.write(destination_backup)

        raise e
