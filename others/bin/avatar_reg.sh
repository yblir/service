#!/bin/bash

netstat -ntpl|grep "[0-9]:${SERVER_PORT} " -q;
if [ $? != 1  ]
then
  echo "端口${SERVER_PORT}已经启用"
  hostip=`ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addrs"`
  #判断注册中心实例是否健康，不健康则重新注册

  OLD_IFS="$IFS"
  IFS=","
  group_arr=($LB_GROUP_NAME)
  IFS="$OLD_IFS"
  for group in ${group_arr[@]}
  do
    stat=`curl -s "http://${AVATAR_LB_URL}/nacos/v1/ns/instance?serviceName=${SERVER_NAME_IN_LB}&groupName=${group}&ip=${hostip}&port=${SERVER_PORT}" | awk -F : '{print $7}'| awk -F , '{print $1}'`
    if [ "true" != "$stat" ]; then
	  curl -X POST -s "http://${AVATAR_LB_URL}/nacos/v1/ns/instance?serviceName=${SERVER_NAME_IN_LB}&ip=${hostip}&port=${SERVER_PORT}&groupName=${group}&weight=${WEIGHT}&ephemeral=false"
    fi
  done
fi
