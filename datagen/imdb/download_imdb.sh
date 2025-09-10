#!/bin/bash
set -ex

FILE="imdb.tgz"

# Create the directory 'datasets/job' if it doesn't already exist,
# then change the current working directory to 'datasets/job'.
mkdir -p data && pushd data
if [ ! -f "imdb.tgz" ]; then
    wget -c https://event.cwi.nl/da/job/imdb.tgz  -O "imdb.tgz"
fi

# Extract only the required files
tar -xvzf imdb.tgz \
    cast_info.csv \
    movie_companies.csv \
    movie_info.csv \
    movie_info_idx.csv \
    movie_keyword.csv \
    title.csv

popd

uv run add_headers_from_sql.py --dir data
uv run convert_to_para.py