"""Random, neutral names for font databases (not derived from the subtitle they were learned from)."""
from __future__ import annotations

import secrets
from pathlib import Path

ADJECTIVES = (
    "amber azure bold brisk calm crisp dusky eager fair fleet gentle glossy golden hazy humble ivory "
    "jolly keen lively lucid mellow misty noble olive pale plain quiet rapid rosy rustic sandy sharp "
    "silent silver sleek slim smooth snowy solar sober spry steady still stout sunny swift tame tidy "
    "urban vivid warm wary wild windy wise witty young zesty agile brave clear deft fresh grand"
).split()
NOUNS = (
    "falcon otter heron lynx badger raven marten ibis puffin gecko bison crane egret ferret finch hare "
    "kestrel koala lark lemur magpie mole newt ocelot osprey panda pelican plover quail robin sable "
    "seal shrike skink sparrow stoat swift tapir tern thrush toucan trout viper vole walrus weasel wren "
    "yak zebra alder aspen birch cedar elm hazel larch maple oak pine rowan spruce willow yew fern"
).split()


def random_db_name(db_dir: Path | None = None) -> str:
    """e.g. 'amber-falcon-3f2a' (~16M combinations); avoids names already used in db_dir."""
    while True:
        name = f"{secrets.choice(ADJECTIVES)}-{secrets.choice(NOUNS)}-{secrets.token_hex(2)}"
        if db_dir is None or not any(db_dir.glob(f"{name}*.json")):
            return name
