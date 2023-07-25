 export PYTHONPATH="${PYTHONPATH}:/"
 export $(grep -v '^#' .env | xargs)
 export $(grep -v '^#' .env-local-no-docker | xargs)
