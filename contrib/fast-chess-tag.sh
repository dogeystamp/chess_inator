#!/bin/sh
# Runs a fast-chess (https://github.com/Disservin/fastchess) tournament based
# on two tags of the chess_inator (https://github.com/dogeystamp/chess_inator)
# engine.
#
# Example usage:
#
#	 cd chess_tournaments
#	 fast-chess-tag.sh quiescence no-quiescence -openings file=8moves_v3.pgn format=pgn order=random -each tc=8+0.08 -each option.Hash=200 -rounds 1200 -repeat -concurrency 8 -recover -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05
#
# You need to be in a chess_inator Git repository to run this script. Ensure
# that the repository you're in is a throw-away worktree. Create one using:
#
# 	git worktree add ../chess_tournaments
#
# inside the original chess_inator repo.
#
# To create a git tag at the current commit, run
#
# 	git tag [tag_name]
#
# or to optionally annotate the tag,
#
# 	git tag -a [tag_name]
#
# which will prompt for a message.
#
# Also, get an opening book from Stockfish's books:
#
# 	curl -O https://github.com/official-stockfish/books/raw/refs/heads/master/8moves_v3.pgn.zip
#
# The sprt mode is a statistical hypothesis testing mode that will tell you how
# probably the first tag is better than the second tag. To check that the
# engine hasn't had a regression, set the Elo bounds to them to [-10, 0]. To
# check for an improvement, use [0, 10]. Alpha and beta are probabilities for
# statistical errors. The tournament automatically ends when a statistically
# significant result is obtained.
#
# LLR stands for "log likelihood ratio", and when it reaches one of the bounds,
# the SPRT concludes. A negative LLR indicates that your hypothesis is wrong,
# while a positive indicates your hypothesis (usually that your new version is
# better, or not worse than the last one) is right. If the LLR is between the
# bounds when the match ends, then the result is not statistically significant.
# This means you need a larger sample size.
#
# By default, a PGN file will be exported with the games played, and the
# fast-chess SPRT output will be appended. This comment may interfere with
# importing the PGN. But Lichess will ignore it, so it's probably fine.

set -e

TAG1="$1"
TAG2="$2"

# if this line fails it's because you don't have enough arguments
shift 2

COMM1=$(git rev-parse --short "$TAG1")
COMM2=$(git rev-parse --short "$TAG2")

mkdir -p games

PGN=games/"$TAG1"__"$TAG2".pgn

rm -f engine1 engine2

IDX=1
while [ -e "$PGN" ]; do
	PGN=games/"$TAG1"__"$TAG2"__"$IDX".pgn
	IDX=$(( $IDX + 1 ))
done
printf "using pgn output: %s\n" "$PGN" > /dev/stderr

git checkout "$TAG1"
cargo build --release
cp target/release/chess_inator engine1

git checkout "$TAG2"
cargo build --release
cp target/release/chess_inator engine2

OUTPUT=$(mktemp)

fastchess \
	-engine cmd=engine1 name="c_i $TAG1 ($COMM1)" \
	-engine cmd=engine2 name="c_i $TAG2 ($COMM2)" \
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
