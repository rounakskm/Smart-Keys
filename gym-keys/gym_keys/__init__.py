#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:59:49 2018

@author: Soumya
"""
# Importing Libraries
from gym.envs.registration import register

# Register the training env in the gym, inorder to be able to instanciate it
red = register(
        id = 'keys-v0',
        entry_point = 'keys_env:KeysEnv',   # filename:ClassName
        timestamp_limit = 100000)           # timelimit for one episode

