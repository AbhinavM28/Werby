.PHONY: install run frontend test lint format docker-up

install:          ## Install backend + dev + frontend deps
	pip install -e ".[dev,frontend]"

run:              ## Run the API with hot reload
	uvicorn app.main:app --reload --port 8000

frontend:         ## Run the Streamlit UI (API must be running)
	streamlit run frontend/streamlit_app.py

test:             ## Run the unit test suite
	pytest -v

lint:             ## Static checks
	ruff check app tests scripts && mypy app

format:           ## Auto-format and fix imports
	ruff check --fix app tests scripts && ruff format app tests scripts

docker-up:        ## Build and start the containerized API
	docker compose up --build
