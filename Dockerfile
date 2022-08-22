FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y texlive-latex-extra
RUN apt-get -y update && apt-get install -y dvipng
RUN apt-get -y update && apt-get install -y ghostscript

RUN apt-get -y update && apt-get install -y mesa-utils
RUN apt-get -y update && apt-get install -y gtk+3.0
RUN apt-get -y update && apt-get install -y gir1.2-gtk-3.0
ENV NO_AT_BRIDGE=1

RUN apt-get -y update && apt-get install -y python3-pip

RUN apt-get -y update && apt-get install -y python3-gi
RUN apt-get -y update && apt-get install -y python3-gi-cairo

RUN apt-get -y update && apt-get install -y gnupg
RUN apt-get -y update && apt-get install -y wget
RUN echo "deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25
RUN apt-get -y update && apt-get -y install python3-graph-tool

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

RUN pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cpu.html