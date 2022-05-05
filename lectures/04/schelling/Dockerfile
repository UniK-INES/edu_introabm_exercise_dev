FROM python:3.9-slim-bullseye

ENV PORT 8521
RUN apt-get update
RUN apt-get install subversion -y
RUN apt-get install git -y

RUN mkdir /mesa
WORKDIR /mesa

ARG FETCH_REQUIRMENTS=2016-01-01
RUN svn checkout https://github.com/UniK-INES/mesa/branches/ines_edu/examples/schelling

RUN pip install -r ./schelling/requirements.txt

ARG FETCH_MODEL=FALSE
RUN svn checkout https://github.com/UniK-INES/mesa/branches/ines_edu/examples/schelling

CMD python ./schelling/run.py