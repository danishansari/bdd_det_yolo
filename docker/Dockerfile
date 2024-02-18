# Dockerization of Yolo BDD-detection model pipeline
FROM ubuntu:22.04

WORKDIR /yolo-bdd

# Poetry's configuration:
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' 

RUN echo 'alias python="python3"' >> ~/.bashrc

RUN apt-get update

RUN apt-get install -y \
    ffmpeg libsm6 libxext6 \
    curl python3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY pyproject.toml /yolo-bdd/
COPY poetry.lock /yolo-bdd/

RUN poetry install -v

CMD ["bash"]
