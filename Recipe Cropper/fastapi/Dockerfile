FROM python:3.12.1-bullseye

# Copy only requirements to cache them in docker layer
COPY pyproject.toml .
COPY poetry.lock .

ENV PYTHONPATH=/
# System deps:
ENV POETRY_VERSION=1.7.1
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false --local
RUN poetry install

# Copy rest of the code
COPY utils/ ./utils/
COPY oma_recipecropper/ ./oma_recipecropper/

EXPOSE 8000
