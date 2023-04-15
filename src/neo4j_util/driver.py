from functools import lru_cache
import logging
from neo4j import GraphDatabase

from util import config_utils

USER = "neo4j"


def get_driver():
    args = config_utils.get_args()
    host = config_utils.get_neo4j_host()
    password = config_utils.get_neo4j_password()
    driver = GraphDatabase.driver(host, auth=(USER, password))
    return driver
