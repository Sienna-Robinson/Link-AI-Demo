# Link-AI-Demo
![Static Badge](https://img.shields.io/badge/version-0.4-pink)
## Table of Contents
### 1. Description
### 2. Installation
- Clone the repository
- Create a virtual environment
- Install dependencies
- Aquire API keys
- Add API keys to the .env file
### 3. Usage



## Description
### This project was created to demonstrate the possibility of having a helpful AI assistant integrated within Link Engine Management's companion app. The AI assistant is to provide correct, safe, and helpful information to the user with whatever queries they may have. 
### Utilising techniques such as tooling, RAG techniques, deterministic safety checking, and LLM routing, this demo is sufficient in answering questions already, and shows proof of concept. The real implementation will include all of these features, along with an AI agent to coordinate everything, an MCP server, and an LLM safety check of the initial prompt and the final draft response. 
### This will ensure complete safety of the user and company, along with correct and useful information that the user may want.

## Installation
### 1. Clone the repository:
```bash
git clone https://github.com/Sienna-Robinson/Link-AI-Demo.git
cd Link-AI-Demo
```

### 2. (Optional) Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```
#### The creator used conda for generating a virtual environment instead of venv, but both are acceptable.
### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Aquire API keys:
#### This project requires access to at least one (preferably two) LLM models. The demo used GPT-4.1 mini for the small, fast model, and Claude 4.5 Opus for the big, higher-reasoning model. This is done using environment variables.
#### An example file was provided, please copy this:
```bash
cp .env.example .env
```
#### To get API keys, search the provider that you wish to use + 'API key', i.e.
```google
OpenAI API key
```
#### It will prompt you to sign in/sign up. You can then create an API key, add a payment method, and copy that API key. 

### 5. Add API keys to .env file:
#### Don't share this key with anyone.
#### Add this key in between the double quotes ("") in the .env file you just copied. 
#### Ensure the name of the key matches the provider. If you must change this (optional), you must change the name in the brackets of api_key=os.getenv() in files:
- build_index.py (safety model)
- retiever.py (safety model)
- llm_router.py (safety model)
- synthesizer.py (brain model)

## Usage
### When you wish to load the server to test the companion, run the following command:
```bash
uvicorn app.main:app --reload --port 8000
```
### This will open up a tab in your browser where the UI will load.
- On the left, there is the chatbot where you can start typing your queries immediately. 
- On the right is a trace window which will show structured JSON data for each query you send.

### Notes:
#### The fault code JSON used for tooling data is currently ChatGPT generated and aren't specific to Link. It is just an example.
#### All other data provided is Link-specific.