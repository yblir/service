#!/bin/bash

source ./image-name-tag.sh
if [ "A$IMAGE NAME" == "A" ] ;then echo "ERROR:Image name not defined.Define the image name in [image-name-tag.gh] and try again";exit -1;fi
if [ "A$IMAGE TAG" == "A" ] ;then echo "ERROR:Image tag not defined.Define the image tag in file (image-name-tag.sh] and try again";exit -1;fi

#Should be moved into select section.
echo "Cleaning old image"
rm -rf image/*
cp image-name-tag.sh ./bin

for i in `ls ./base_image/`;do echo "Loading base image...";docker load <./base_image/$i;done
echo "Building image [$IMAGE_NAME:$IMAGE TAG]..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .

if [ $? -ne 0 ] ; then echo "ERROR: Failed to build image";exit -1;fi

echo "Do you want to save the newly built image [$IMAGE_NAME: $IMAGE_TAG] ?"
read -p "Please input (y/n): " yn
if [ "$yn" == "Y" ] || [ "$yn" == "y" ]; then
    echo "Saving image..."; docker save $IMAGB_NAME:$IMAGE_TAG > ./image/${IMAGE_NAME}_${IMAGE_TAG}.tar;echo "The image has been saved.";
else
    echo "The image will not be saved.";
fi