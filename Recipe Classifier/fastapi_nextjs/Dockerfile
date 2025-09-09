FROM python:3.12.1-bullseye

# Add NodeSource repo for Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
# Install Node.js and npm
RUN apt-get update && apt-get install -y nodejs

# Copy only requirements to cache them in docker layer
COPY pyproject.toml .
COPY poetry.lock .

ENV PYTHONPATH=/
# System deps:
ENV POETRY_VERSION=1.8.3
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false --local
RUN poetry install

# Copy rest of the code
COPY utils/ ./utils/
COPY oma_recipeclassifier/ ./oma_recipeclassifier/

# Build Next.js app
RUN npm install --prefix ./oma_recipeclassifier/demo
RUN npm run build --prefix ./oma_recipeclassifier/demo

EXPOSE 8000
EXPOSE 3000
