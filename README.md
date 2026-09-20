# MedQuery AI

**Build a Complete Medical Intelligent Health Companion using LLMs, LangChain, Pinecone, Streamlit & AWS**

---

## Overview

MedQuery AI is an intelligent medical assistant designed to provide reliable and concise answers to healthcare-related questions.
It uses **Large Language Models (LLMs)** integrated with **LangChain**, **Groq**, and **Pinecone** to process medical documents, generate embeddings, and deliver accurate query-based responses.
This project demonstrates how AI and modern NLP techniques can assist in medical information retrieval, using a lightweight and modular architecture.

---

## Features

* Context-aware medical question answering
* PDF document parsing and knowledge embedding
* Fast semantic search powered by **Pinecone Vector Database**
* Flask backend for query processing
* Streamlit web interface for easy interaction
* Secure environment variable management using `.env`

---

## Project Structure

```
MEDQUERY_AI/
│
├── data/                     # Data files or PDFs (knowledge source)
├── research/                 # Jupyter notebooks and research experiments
│   └── main.ipynb
├── src/                      # Core application logic
│   ├── __init__.py
│   ├── helper.py
│   └── prompt.py
│
├── app.py                    # Flask application
├── streamlit.py              # Streamlit web interface
├── store_index.py            # Embedding and Pinecone index creation
├── requirements.txt          # Project dependencies
├── setup.py
├── .env                      # Environment variables
└── README.md
```

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MedQuery_AI.git
cd MedQuery_AI
```

### 2. Create and activate the environment

```bash
conda create -n medicalai python=3.10 -y
conda activate medicalai
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


## Running the Application

### 1. Store document embeddings to Pinecone

```bash
python store_index.py
```

### 2. Start the backend server

```bash
python app.py
```

### 3. Launch the Streamlit interface

```bash
streamlit run streamlit.py
```

Now, open your browser and navigate to:

```
http://localhost:8501
```

---

## Model & Index Details

* **Groq Model:** `llama-3.3-70b-versatile`
* **Pinecone Index Name:** `medical-queryai`
* **Conda Environment:** `medicalai`

---

## Tech Stack

* **Python**
* **LangChain**
* **Groq LLMs**
* **Pinecone**
* **Streamlit**
* **Flask**
* **AWS**

---



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI:892825672696.dkr.ecr.us-east-1.amazonaws.com/medquery_ai
	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - PINECONE_API_KEY
   - GROQ_API_KEY
