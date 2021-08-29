FROM continuumio/anaconda3:latest

RUN apt-get update
RUN apt-get install -y build-essential wget vim

RUN conda install -y setuptools pystan && \
    conda install -y -c conda-forge arviz
RUN conda update --all

RUN conda list ipython

ENV APP_ROOT /app
WORKDIR $APP_ROOT

ADD . $APP_ROOT

RUN ipython profile create
RUN echo "c.InteractiveShellApp.exec_lines = ['import sys; sys.path.append(\"/app\"); sys.path.append(\"/app/utils\"); sys.path.append(\"/app/OASIS\"); sys.path.append(\"/app/OASIS/oasis\")']" >> /root/.ipython/profile_default/ipython_config.py
