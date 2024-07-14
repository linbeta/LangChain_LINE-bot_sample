FROM python:3.12-slim

RUN pipx install poetry==1.8.3

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./package[s] ./packages

RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

COPY ./rag ./rag

RUN poetry install --no-interaction --no-ansi

EXPOSE 8080

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080
