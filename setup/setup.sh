#!/bin/bash

conda create --name sic python=3.9.21

conda run -n sic pip install -r setup/requirements.txt