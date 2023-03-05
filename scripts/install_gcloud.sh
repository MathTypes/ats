# download the gcloud tar.gz 
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-316.0.0-darwin-x86_64.tar.gz
# extract the file
tar -xf google-cloud-sdk-316.0.0-darwin-x86_64.tar.gz
# install gloud
./google-cloud-sdk/install.sh
# initialize gcloud
./google-cloud-sdk/bin/gcloud init
