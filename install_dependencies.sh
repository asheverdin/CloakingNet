#!/usr/bin/env bash
echo "Please install conda if it is not installed!"
echo "Installing the environment:"
conda env create -f environment.yml
conda activate cloaking_net