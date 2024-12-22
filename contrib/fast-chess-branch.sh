#!/bin/sh
# Runs a fast-chess (https://github.com/Disservin/fastchess) tournament based
# on two branches of the chess_inator
# (https://github.com/dogeystamp/chess_inator) engine.
#
# Example usage:
#
#	 cd chess_tournaments
#	 fast-chess-branch.sh quiescence no-quiescence -openings file=8moves_v3.pgn format=pgn order=random -each tc=300+0.1 -rounds 12 -repeat -concurrency 8 -recover -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05
#
# Do not use `main` as a branch, or any other branch already checked out in
# another directory. You need to be in a chess_inator Git repository to run
# this script. Ensure that the repository you're in is a throw-away worktree.
# Create one using
#
# 	git worktree add ../chess_tournaments
#
# inside the original chess_inator repo.
# Also, get an opening book from Stockfish's books:
#
# 	curl -O https://github.com/official-stockfish/books/raw/refs/heads/master/8moves_v3.pgn.zip
#
# The sprt mode is a statistical hypothesis testing mode that will tell you how
# probably the first branch is better than the second branch. The Elo ratings
# given are the "indifference zone" where the result is acceptable. To check
# that the engine hasn't had a regression, set them to [-10, 0]. To check for
# an improvement, use [0, 10]. Alpha and beta are probabilities for statistical
# errors. The tournament automatically ends when a statistically significant
# result is obtained.
#
# By default, a PGN file will be exported with the games played, and the
# fast-chess SPRT output will be appended. This comment may interfere with
# importing the PGN. But Lichess will ignore it, so it's probably fine.

set -e

BRANCH1="$1"
BRANCH2="$2"

# if this line fails it's because you don't have enough arguments
shift 2

COMM1=$(git rev-parse --short "$BRANCH1")
COMM2=$(git rev-parse --short "$BRANCH2")

mkdir -p games

PGN=games/"$BRANCH1"__"$BRANCH2".pgn

rm -f engine1 engine2
if [ -f "$PGN" ]; then
	rm -i "$PGN"
fi

git switch "$BRANCH1"
cargo build --release
cp target/release/chess_inator engine1

git switch "$BRANCH2"
cargo build --release
cp target/release/chess_inator engine2

OUTPUT=$(mktemp)

fastchess \
	-engine cmd=engine1 name="c_i $BRANCH1 ($COMM1)" \
	-engine cmd=engine2 name="c_i $BRANCH2 ($COMM2)" \
	-pgnout file="$PGN" \
	timeleft=true \
	$@ \
	2>&1 | tee -a "$OUTPUT"

printf "\n{" >> "$PGN"

# match between ------- markers in fastchess output, strip newline and then output to PGN
awk '/-{50}/{f+=1; print; next} f%2' "$OUTPUT" \
	| head -c -1 \
	>> "$PGN"

printf "}" >> "$PGN"

rm "$OUTPUT"
