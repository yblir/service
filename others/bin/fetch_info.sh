#!/bin/bash


function is_process_on()
{
  # aiserver check
  sign=1
  for k in $( seq 1 ${INSTANCE_NUM} )
  do

        #判断端口是否被占用
        portk=$((${SERVER_PORT} + $k))
        netstat -ntpl|grep "[0-9]:${portk} " -q;
        if [ $? == 1  ]
        then
          sign=0
        fi
  done

  if [ $sign == 1 ]
  then
    echo "aiservice:running"
  else
    echo "aiservice:stopped"
  fi
  # nginx service check
  ps -few | grep nginx | grep master -q;
  if [ $? != 1  ]
  then
    echo "nginx:running"
  else
    echo "nginx:stopped"
  fi

}

is_process_on
###########################################################################

exit 0
#end
