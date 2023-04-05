import argparse

args = None

DEFAULT_HOST = 'bolt://10.0.0.18:7687'
# host = 'bolt://host.docker.internal:7687'
USER = 'neo4j'
DEFAULT_PASSWORD = 'password'

def set_args(new_args):
    global args
    args = new_args

def get_args():
    global args
    return args

def get_arg_parser(description):
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--neo4j_host", type=str, required=False)
    parser.add_argument("--neo4j_password", type=str, required=False)
    return parser

def get_neo4j_host():
    args = get_args()
    host = DEFAULT_HOST
    if args and args.neo4j_host:
        host = args.neo4j_host
    return host

def get_neo4j_password():
    args = get_args()
    password = DEFAULT_PASSWORD
    if args and args.neo4j_password:
        password = args.neo4j_password
    return password

