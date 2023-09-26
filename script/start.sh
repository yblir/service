#!/bin/bash

# 服务所在基础目录
app_dir_name="face_attribute_acc"
# 服务启动命令, 有service端口监控那个
app_run_cmd="python3 face_attribute_servce.py"

# 将模型文件拷贝到公共区域，各个子示例调用相同的模型文件，提升微服务启动的速度
mv  /home/${app_dir_name}/models /home/models
sed -i "s#=\s*models#=/home/models#g"  /home/${app_dir_name}/config/config.yaml

# 将微服务的配置参数以环境变量的配置参数形式映射进去
for k in $( seq 1 ${INSTANCE_NUM} )
do
  # 拷贝多份服务目录
  cp -r /home/${app_dir_name}/ /home/${app_dir_name}$k

  # 修改各个服务配置
  CONFIG_FILE=/home/${app_dir_name}$k/config/config.yaml
  portk=$((${SERVER_PORT} + $k))
  sed -i "s#device_id\s*=.*#device_id=${GPU_IDS}#g" ${CONFIG_FILE}
  sed -i "s#server_port\s*=.*#server_port=${portk}#g" ${CONFIG_FILE}
#  sed -i "s#cpu_num\s*=.*#cpu_num=${CPU_NUM}#g" ${CONFIG_FILE}
#  sed -i "s#conf\s*=.*#conf=${CONF}#g" ${CONFIG_FILE}
#  sed -i "s#iou\s*=.*#iou=${IOU}#g" ${CONFIG_FILE}
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
GUARD_FILE=/root/.docker/agent/guard.xml

# 预先开启实例生成引擎，减少显存占用
app_path=/home/${app_dir_name}1
cd ${app_path}
# 启动服务
nohup ${app_run_cmd} > ai.log  2>&1 &
sleep 10

# 等待启动完成
while true
do
    #通过端口查看服务是否已经启动完成
    portk=$((${SERVER_PORT} + 1))
    netstat -ntpl|grep "[0-9]:${portk} " -q;
    if [ $? != 1  ]
    then
        break
    else
        sleep 10
    fi
done
killall python3
# rm ai.log

# 逐个服务进程启动
for k in $( seq 1 ${INSTANCE_NUM} )
do

  app_path=/home/${app_dir_name}$k
  cd ${app_path}

  # 清空日志
  # rm -rf log/*  >/dev/null  2>&1

  # 启动服务
  nohup ${app_run_cmd} >/dev/null  2>&1 &
  sleep 10

  # 等待启动完成
  while true
  do

      #判断端口是否被占用
      portk=$((${SERVER_PORT} + $k))
      netstat -ntpl|grep "[0-9]:${portk} " -q;
      if [ $? != 1  ]
      then
          break
      else
          sleep 10
      fi
  done

done

# nginx服务启动
sed -i "s#worker_processes  5#worker_processes  ${INSTANCE_NUM}#g" /usr/local/nginx/conf/nginx.conf
# 替换nginx对外的端口
sed -i "s#listen       80;#listen       ${SERVER_PORT};#g" /usr/local/nginx/conf/nginx.conf
for k in $(seq 1 ${INSTANCE_NUM} )
do
  portk=$((${SERVER_PORT} + $k))

  sed -i "40i \ \ \ \ server localhost:${portk};" /usr/local/nginx/conf/nginx.conf

done

cd /usr/local/nginx/sbin
./nginx
/usr/local/nginx/sbin/nginx -s reload


# 启动守护程序
sed -i 's#Enable="false"#Enable="true"#g' ${GUARD_FILE}
/usr/bin/systemctl restart cobeagent
echo "start finished" > ai.log
