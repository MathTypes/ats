PYTHONPATH=. python3 twitter/download_tweet.py --stock=$3 --since=$1 --until=$2 --neo4j_host=bolt://host.docker.internal:7687
