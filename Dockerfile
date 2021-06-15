FROM continuumio/anaconda3

ENV APP_ROOT /app
RUN mkdir $APP_ROOT
WORKDIR $APP_ROOT

ADD . $APP_ROOT

RUN pip install --upgrade setuptools && pip install -r requirements.txt

RUN ipython profile create
RUN echo "c.InteractiveShellApp.exec_lines = ['import sys; sys.path.append(\"/app\"); sys.path.append(\"/app/utils\")']" >> /root/.ipython/profile_default/ipython_config.py
