#!/bin/bash

GEN_ROOT=$(realpath $(dirname $0))
mkdir $GEN_ROOT/simulation-trial2
cp $GEN_ROOT/simulation-trial1/driver.sh $GEN_ROOT/simulation-trial2/driver.sh
mkdir $GEN_ROOT/train-trial2
cp $GEN_ROOT/train-trial1/driver.sh $GEN_ROOT/train-trial2/driver.sh
cp $GEN_ROOT/train-trial1/game-trial1.pth.4 $GEN_ROOT/train-trial2/game-trial2.pth

container_ids=()
for i in {1..3}; do
    container_id=$(docker run --rm -u alankao -w /home/alankao/PathogenEngine -v $GEN_ROOT/..:/mnt -d pathogen:base-cpu bash /mnt/$(basename $GEN_ROOT)/simulation-trial2/driver.sh 2)
    container_ids+=("$container_id")
    echo "Started container $container_id"
done

# Wait for all containers to stop
for container_id in "${container_ids[@]}"; do
    docker wait "$container_id"
    echo "Simulation Container $container_id has stopped."
done

python simulation-trial2/collect.py simulation-trial2 simulation-trial2/game.bin
python simulation-trial2/dataset.py simulation-trial2/game.bin 

container_id=$(docker run --rm -u alankao -w /home/alankao/PathogenEngine -v $GEN_ROOT/..:/mnt -d pathogen:base bash /mnt/$(basename $GEN_ROOT)/train-trial2/driver.sh 2)
echo "Started container $container_id"

docker wait "$container_id"
echo "Training Container $container_id has stopped."


