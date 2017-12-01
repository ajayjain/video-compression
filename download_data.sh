#!/usr/bin/env bash

# Download uncompressed videos from the LIVE database
LIVE_BASE_URL="http://vqdatabase.ece.utexas.edu/research/live/vqdatabase/videos/"

read -p "LIVE user: " live_user
read -s -p "LIVE password: " live_password
echo ""

mkdir -p data/live

live_videos=(bs1_25fps.yuv mc1_50fps.yuv pa1_25fps.yuv rb1_25fps.yuv rh1_25fps.yuv sf1_25fps.yuv st1_25fps.yuv tr1_25fps.yuv)
for video in "${live_videos[@]}"; do
    prefix="${video:0:2}"
    url="$LIVE_BASE_URL$prefix Folder/$video"

    echo "[INFO] Downloading $url"
    wget "$url" --user="$live_user" --password="$live_password" -P data/live/
    echo ""
done

live_videos=(pr1_50fps.yuv sh1_50fps.yuv)
for video in "${live_videos[@]}"; do
    prefix="${video:0:2}"
    url="$LIVE_BASE_URL${prefix}1 Folder/$video"

    echo "[INFO] Downloading $url"
    wget "$url" --user="$live_user" --password="$live_password" -P data/live/
    echo ""
done

wget http://vqdatabase.ece.utexas.edu/research/live/vqdatabase/readme.txt --user="$live_user" --password="$live_password" -P data/live/
