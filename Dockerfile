#第一部分
FROM ai-nj-sailor:cuda11.6.2-cudnn8-trt8.2.4.2-runtime
LABEL author="X8149"\
      description="face attribute acc service"

USER root

#此行不需要修改
WORKDIR /home
RUN echo "/root/.docker/agent/cobeagent -f /root/.docker/agent/guard.xml" >> /etc/rc.local

#第二部分
######install sailor pkgs############
ARG CONF_PATH=/root/.docker/
ADD script/* ${CONF_PATH}
ADD agent/* ${CONF_PATH}/agent/
RUN /usr/bin/dos2unix /root/.docker/agent/guard.xml;
RUN /usr/bin/dos2unix /root/.docker/*.sh;chmod a+x /root/.docker/*.sh;

#第三部分
ARG workdir=/home
ADD package/face attribute acc ${workdir}/face_attribute_acc
ADD package/nginx /usr/local/nginx

RUN chmod +x /usr/local/nginx/sbin/nginx

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video, compute,utility

#微服务框架必要的参数，不需要改变
ENV SERVER_PORT=8080

#微服务框架必要的参数，根据服务进行修改
ENV GPU_IDS=0
ENV INSTANCE_NUM=4

#根据各算法服务不同，进行修改
#ENV BATCHSIZE=32

#此行不需要修改
ENTRYPOINT ["/etc/rc.d/rc.local"]