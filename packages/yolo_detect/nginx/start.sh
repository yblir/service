#! /bin/sh
cd sbin
./nginx -s stop > log.log
sleep 2
./nginx
cd ..