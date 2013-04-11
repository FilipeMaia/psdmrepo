"""Setup the icws application"""
import logging

from icws.config.environment import load_environment

log = logging.getLogger(__name__)

def setup_app(command, conf, vars):
    """Place any commands to setup icws here"""
    load_environment(conf.global_conf, conf.local_conf)
