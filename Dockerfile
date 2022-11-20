FROM matthewfeickert/docker-python3-ubuntu

USER root

WORKDIR /InformationSystemProject

ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 python3-pip  -y

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools

COPY . ./

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirement.txt
RUN rm -rf ~/.cache/pip

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]