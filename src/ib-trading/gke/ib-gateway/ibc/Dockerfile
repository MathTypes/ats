FROM akolotech/ibc:v0
ARG TWS_USERID=tws_userid
ARG TWS_PASSWORD=tws_password

ENV JAVA_VRSN=1.8.0_202 \
  TWS_MAJOR_VRSN=1019 \
  TRADING_MODE=live \
  TWSUSERID=${TWS_USERID} \
  TWSPASSWORD=${TWS_PASSWORD}

# set environment variables
ENV APP=GATEWAY \
    IBC_INI=/root/ibc/config.ini \
    IBC_PATH=/opt/ibc \
    JAVA_PATH_ROOT=/opt/i4j_jres \
    LOG_PATH=/opt/ibc/logs \
    TWS_INSTALL_LOG=/root/Jts/tws_install.log \
    TWS_PATH=/root/Jts \
    TWS_SETTINGS_PATH=/root/Jts

# make dirs
RUN mkdir -p /tmp && mkdir -p ${IBC_PATH} && mkdir -p ${TWS_PATH}

# copy IBC/Jts configs
# For some reason, the copy does not seem to overwrite those files. End up committing
# container with the right config files.
#COPY config/config.ini ${IBC_INI}
#COPY config/jts.ini ${TWS_PATH}/jts.ini

# copy cmd script
WORKDIR /home
COPY cmd.sh cmd.sh
RUN chmod +x cmd.sh

# set display environment variable (must be set after TWS installation)
ENV DISPLAY=:0

# execute cmd script
ENTRYPOINT "/home/cmd.sh"
