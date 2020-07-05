#!/usr/bin/env python3.7
"""
Integration test script

Usage:
    python3.7 -m py.test
"""


import pytest
import glob
import time
import requests
import os

model_upload = '/apps/models/*'

def test_backend():
    files = list(glob.glob(model_upload))
    assert len(files) == 3

def test_frontend():
    try:
        response = requests.get('http://frontend:5000/files')
        assert response.status_code == 200
    except:
        assert False
