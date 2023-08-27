#!/bin/bash

if [ $# != 1 ];
then
    echo "Usage: sh aiservice_check.sh [url]"
	exit 1
fi

url=$1

echo "curl --location --request POST "${url}" --header "Content-Type: application/json" -d @data.json"

curl --location --request POST "${url}" --header "Content-Type: application/json" -d @data.json

echo ""