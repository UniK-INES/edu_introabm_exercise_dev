FROM python:3.9-slim-bullseye

ENV PORT 8522
RUN apt-get update
RUN apt-get install subversion -y
RUN apt-get install git -y

RUN mkdir /mesa
WORKDIR /mesa

ARG FETCH_REQUIRMENTS=2016-01-01
RUN svn checkout https://github.com/UniK-INES/edu_introabm_exercise_dev/trunk/abmodel
RUN pip install -r ./abmodel/requirements_abmodel.txt

ARG FETCH_MODEL=FALSE
RUN svn checkout https://github.com/UniK-INES/edu_introabm_exercise_dev/trunk/abmodel
#RUN pip install -I git+https://github.com/UniK-INES/mesa@ines_edu

CMD python ./abmodel/run.py