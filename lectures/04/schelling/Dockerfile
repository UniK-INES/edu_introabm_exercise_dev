FROM python:3.9-slim-bullseye

ENV PORT 8521
RUN apt-get update
RUN apt-get install subversion -y
RUN apt-get install git -y

RUN mkdir /mesa
WORKDIR /mesa
RUN svn checkout https://github.com/UniK-INES/mesa/branches/ines_edu/examples/schelling

RUN pip install -r ./schelling/requirements.txt
RUN echo reinstall
#RUN pip install -I git+https://github.com/UniK-INES/mesa@ines_edu

CMD python ./schelling/run.py