#!/bin/bash

python3 -m venv venv

source venv/bin/activate

pip3 install -r CW1-60012/requirements.txt

echo "Virtual environment setup and package installation completed."
