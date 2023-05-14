FROM gcr.io/keen-rhino-386415/akolotech/ibc/cloudrun_base:v1.3

# copy Python application and pip install packages
WORKDIR /home
COPY app .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# execute cmd script
CMD ./cmd.sh
