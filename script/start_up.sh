#!/bin/bash

# 服务所在基础目录
app_dir_name="face_attribute_acc"
# 服务启动命令, 这两条指令同start.sh
app_run_cmd="python3 face_attribute_service.py"

# 将微服务的配置参数以环境变量的配置参数形式映射进去
for k in $( seq 1 ${INSTANCE_NUM} )
do

  # 修改各个服务配置
  CONFIG_FILE=/home/${app_dir_name}$k/config/app.conf
  portk=$((${SERVER_PORT} + $k))
  sed -i "s#device_id\s*=.*#device_id=${GPU_IDS}#g" ${CONFIG_FILE}
  sed -i "s#server_port\s*=.*#server_port=${portk}#g" ${CONFIG_FILE}
  #sed -i "s#batchsize\s*=.*#batchsize=${BATCHSIZE}#g" ${CONFIG_FILE}
  #sed -i "s#fp16_flag\s*=.*#fp16_flag=${USEFP16}#g" ${CONFIG_FILE}
  sed -i "s#post_url\s*=.*#post_url=${COLLECTOR_POST_URL}#g" ${CONFIG_FILE}
  sed -i "s#send_info_time_interval\s*=.*#send_info_time_interval=${SEND_INFO_TIME_INTERVAL}#g" ${CONFIG_FILE}

  #########################
  # 根据不同服务进行参数配置
  #########################

  # # 在schema.json文档中，根据batchsize对微服务的最大batchsize进行修改；
  # sed -i "s#\"maxItems\":.*#\"maxItems\": ${BATCHSIZE}#g" /home/${app_dir_name}$k/config/single_image_schema.json

done

######################################################################################
                           # 默认配置项区(通常不用修改)
######################################################################################

# trt动态库运行需要添加/usr/local/lib64/到环境变量
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64/

for k in $( seq 1 ${INSTANCE_NUM} )
do

  #判断端口是否被占用
  portk=$((${SERVER_PORT} + $k))
  netstat -ntpl|grep "[0-9]:${portk} " -q;
  if [ $? != 1  ]
  then
      echo "端口${portk}已经启用"
  else
      cd /home/${app_dir_name}$k
      rm -rf core.* >/dev/null 2>&1
      nohup ${app_run_cmd} >/dev/null 2>&1 &
      sleep 5
  fi

done

# 重新加载nginx服务
/usr/local/nginx/sbin/nginx -s reload
