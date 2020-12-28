#!/usr/bin/env bash
echo 'Downloading the dataset'
mkdir Data
cd Data || exit
gdown https://drive.google.com/uc?id=1E06aeZXdJ0YEVpbUXQp3gZhE2udNETCX
unzip Data_archive.zip || exit
rm  Data_archive.zip
cd ../
echo "the download is complete