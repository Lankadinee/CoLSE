#!/usr/bin/env bash

docker ps -aq | while read id; do
  size=$(docker inspect -f '{{.SizeRw}}' "$id")
  if [ "$size" -gt 1073741824 ]; then
    echo "Removing container $id (SizeRw = $size bytes)"
    docker rm "$id"
  fi
done

docker builder prune -f --all