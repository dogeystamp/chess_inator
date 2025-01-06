#!/bin/sh

# This file is part of chess_inator.
# chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
#
# chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.
#
# Copyright © 2024 dogeystamp <dogeystamp@disroot.org>


# Filters lichess PGN databases (https://database.nikonoel.fr/) using pgn-extract (https://github.com/MichaelB7/pgn-extract).
#
# - Require 10 plies at least
# - Discard forfeits
# - Require at least 5 minutes time control
#
# Usage:
#
#     s0a_pgn_extract.sh games.pgn -o filtered.pgn

TAGFILE="$(mktemp)"
cat << EOF > "$TAGFILE"
Termination "Normal"
TimeControl >= "300"
EOF

pgn-extract -pl10 -t "$TAGFILE" $@

rm "$TAGFILE"
