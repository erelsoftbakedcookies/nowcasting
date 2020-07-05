#!/bin/bash
for file in `find . -name *.py`; do echo $file; python3.7 -m autopep8 --in-place --aggressive $file; done
