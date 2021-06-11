FROM continuumio/anaconda3

ENV APP_ROOT /app
RUN mkdir $APP_ROOT
WORKDIR $APP_ROOT

ADD . $APP_ROOT

RUN pip install --upgrade setuptools && pip install -r requirements.txt
