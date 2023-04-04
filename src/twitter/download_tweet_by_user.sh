PYTHONPATH=. python3 twitter/download_tweet.py --username=$3 --since=$1 --until=$2 -v  --neo4j_host=bolt://host.docker.internal:7687
