#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import os

def ensure_dir(directory):
    """Make sure the directory exists. If not, create it including subdirs."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
