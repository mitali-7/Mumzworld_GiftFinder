# Mumzworld AI Gift Finder

## Overview

This project is an AI-powered gift recommendation system designed to help users quickly find relevant baby and mom products based on natural language queries.

Instead of just listing products, the system explains *why* a product is a good choice and adds a “mom-to-mom confidence signal” by summarizing real gifting-related reviews.

## What it does

- Takes a natural language query  
  (e.g., “gift for 6-month-old under 100 AED”)

- Understands intent (age, budget, use-case)

- Retrieves relevant products from a catalog

- Filters and uses **gifting-specific reviews**

- Generates structured recommendations with:
  - reasoning
  - “what moms say” insights
  - English + Arabic output

- Handles edge cases:
  - no results
  - out-of-scope queries

## Key Idea

The main addition is a mom's recommendation or review.

Instead of relying on ratings or generic descriptions, the system extracts insights from reviews where people explicitly mentioned gifting the product. This helps users understand *why other moms chose it as a gift*, making recommendations more trustworthy.

## Tech Stack

- **Backend**: FastAPI  
- **LLM**: Groq (LLaMA 3.3 70B)  
- **Retrieval**: TF-IDF   
- **Validation**: Pydantic  
- **Frontend**: Simple HTML + JS  

## How to run

### 1. Setup environment
conda create -n mumz-ai python=3.10
conda activate mumz-ai
pip install -r backend/requirements.txt
pip install requests

### 2. Add API key

Create .env with API key:
GROQ_API_KEY=your_key_here

### 3. Run backend
cd backend
uvicorn main:app --reload

### 4. Open UI
Open:
frontend/index.html


## Evaluation

### Approach

I evaluated the system using a small set of test queries covering:
- normal use cases  
- edge cases (budget constraints)  
- invalid / out-of-scope queries  
- ambiguous inputs  

The goal was to check not just correctness, but also grounding and failure handling.

### What was evaluated

For each query, I checked:

- **Relevance** → Are the recommended products suitable for the query?  
- **Grounding** → Are all outputs coming from the dataset (no hallucination)?  
- **Review usage** → Is "moms_say" based on gifting-related reviews?  
- **Failure handling** → Does the system correctly handle no-results and invalid queries?  
- **Structure** → Is the output valid JSON and consistent?

### Results

All test cases passed under the defined checks.

### Examples:

- "gift for 6 month old under 100 AED" → relevant recommendations  
- "gift under 10 AED" → no results + uncertainty  
- "laptop for coding" → correctly marked as out-of-scope  
- "gift for baby" → broader but still valid recommendations  

### Observations
- The system performs reliably for structured queries  
- Outputs remain grounded in product + review data  
- Failure cases are handled explicitly (no hallucination)

### Limitations
- Very vague queries can lead to generic recommendations  
- Arabic output is functional but not always perfectly natural  
- Retrieval uses TF-IDF, so semantic matching can be improved  


## Tradeoffs
Used TF-IDF instead of embeddings for speed and simplicity
Used synthetic dataset due to scraping constraints
Relied on LLM for multilingual output
Kept frontend minimal to focus on backend system design

## Failure Handling

The system avoids hallucination by:
Returning out_of_scope = true for unrelated queries
Returning empty results + uncertainty message when no match exists
Restricting recommendations strictly to catalog data

## What I would improve next
Replace TF-IDF with embedding-based retrieval
Improve Arabic quality with better post-processing
Add personalization (child age, preferences)
Deploy as a chat-based assistant (e.g., WhatsApp)