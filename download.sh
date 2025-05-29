#!/bin/bash

echo "Running the download script"
if [ ! -d "data" ]; then
    mkdir data
fi

wget -O data/content.zip "https://www.dropbox.com/scl/fo/xr43rau39zpawsmvd65za/AFS27VxCa2gwRsGWEmVsws8?rlkey=ixin86xnfe6r2w61iokr4t5fz&st=knau6fqa&dl=1"

if [ -f data/content.zip ]; then
    echo "Downloaded"
    unzip data/content.zip -d data/
    echo "Unzipped"
    rm data/content.zip
else
    echo "Download failed, data.zip not found!"
fi
