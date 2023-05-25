#!/usr/bin/env bash

# There is a memory leak when running produce_all.jl.
#
# We can run julia garbage collection on every iteration to avoid
# running out of memory, However, this slows down the script by a
# lot. Instead, since the script is idempotent, we can just run it
# multiple times until it succeeds. Since it saves the results to
# disk it won't repeat previous computations.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROG="${1:-"$SCRIPT_DIR/produce_all.jl"}"

MAX_TRIES=10

for i in $(seq 1 $MAX_TRIES); do
    echo "Attempt $i" | notify
    if $PROG;
    then
        echo "Success" | notify
        exit 0
    else
        echo "Failure" | notify
    fi
done
echo "Failure, reached max number of tries ($TRIES)" | notify
