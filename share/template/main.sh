#!/bin/bash

GEN_ROOT=$(realpath $(dirname $0))

container_ids=()
for i in {1..3}; do
    container_id=$(docker run --rm -u alankao -w /home/alankao/PathogenEngine -v $GEN_ROOT/..:/mnt -d pathogen:base-cpu bash /mnt/$(basename $GEN_ROOT)/simulation/driver.sh)
    container_ids+=("$container_id")
    echo "Started container $container_id"
done

# Wait for all containers to stop
for container_id in "${container_ids[@]}"; do
    docker wait "$container_id"
    echo "Container $container_id has stopped."
done

python simulation/collect.py simulation simulation/game.bin
python simulation/dataset.py simulation/game.bin 


