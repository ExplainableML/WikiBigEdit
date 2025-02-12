#!/bin/bash

# Ensure two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <old_wikidata> <new_wikidata>"
    exit 1
fi

OLD_WIKIDATA=$1
NEW_WIKIDATA=$2

# Extract wikidata changes
for IDX in {0..19}; do
    python extract_triplets.py --mode changed --old "$OLD_WIKIDATA" --new "$NEW_WIKIDATA" --idx $IDX --combine 0 --num_workers 10 --batch_size 10000 &
done

# Wait for all jobs to complete
wait
python extract_triplets.py --mode changed --old "$OLD_WIKIDATA" --new "$NEW_WIKIDATA" --idx -1 --combine 1

# Format extracted changes
python process_triplets.py --old "$OLD_WIKIDATA" --new "$NEW_WIKIDATA"

# Run parallel locality probe extraction
for IDX in {0..19}; do
    python process_triplets.py --old "$OLD_WIKIDATA" --new "$NEW_WIKIDATA" --batch_size 20 --loc_dist True --num_splits 20 --split $IDX &
done

# Wait for all locality probe extraction jobs to complete
wait

# Combine locality results
python reorder_loc.py --old "$OLD_WIKIDATA" --new "$NEW_WIKIDATA"

# Generate QA samples
python extract_qa.py --old "$OLD_WIKIDATA" --new "$NEW_WIKIDATA"

# Combine QA sets
python combine_qa.py --old "$OLD_WIKIDATA" --new "$NEW_WIKIDATA"