# Tripletex AI Accounting Agent

AI agent for the AINM Tripletex accounting competition. Receives multilingual accounting tasks via a `/solve` endpoint and executes them against the Tripletex API using Gemini function-calling.

## Current State

| Metric | Value |
|--------|-------|
| **Total Score** | 71.4 |
| **Rank** | #32 |
| **Tasks Attempted** | 30/30 |
| **Submissions** | ~290 |
| **Model** | `gemini-3.1-pro-preview` (AI Studio) |
| **Deployed Rev** | rev58 (accounting-agent-00058) |
| **Cloud Run Region** | europe-west1 |

### Score Progression
- **v1 (rev33)**: ~45 → baseline with gemini-2.5-pro
- **v2 (rev38-40)**: 62.4 → model swap to gemini-3.1-pro-preview + many prompt/handler fixes
- **v3 (rev41-43)**: 62.4 → salary division fix, voucher balance check, bank reconciliation fix
- **v4 (rev46-48)**: 71.4 → date filter fix, empty response handling, rate limit retry, field fixes

### Task Performance
- **Perfect (100%)**: departments, simple suppliers/customers, invoices, orders, products — consistently 7/7 or 8/8
- **Good (60-75%)**: error correction (7.5/10), annual accounts (10/10 sometimes), currency exchange
- **Weak (<40%)**: bank reconciliation (/14), complex onboarding (/10 intermittent), some error correction tasks
- **Known Issue**: Gemini 3.1 Pro Preview has very tight API rate limits (429 RESOURCE_EXHAUSTED), which causes task aborts when submitting multiple tasks or after heavy usage

## Architecture

```
POST /solve
  → main.py (FastAPI)
    → agent.py (Gemini agent loop, max 25 turns)
      → tripletex_client.py (Tripletex REST API client)
```

- **agent.py** (~2250 lines): Gemini function-calling agent with 40+ Tripletex tools, extensive prompt guidance, auto-fix handlers, pre-flight checks
- **main.py**: FastAPI app, extracts task prompt + attachments (PDF/CSV), pre-fetches common entities (accounts, employees, customers, suppliers, VAT types)
- **tripletex_client.py**: HTTP client for Tripletex v2 API with session token management

## Model Configuration (AI Studio vs Vertex AI)

This project now prefers **Google AI Studio API key** access instead of Vertex AI so we can use newer models (for better scores and stability).

- **Primary path**: `GEMINI_API_KEY` (AI Studio) with `gemini-3.1-pro-preview`
- **Fallback**: Vertex AI (if `GEMINI_API_KEY` is missing)

To switch to AI Studio:

1. Add your key to `.env`:

```
GEMINI_API_KEY=your-ai-studio-key
```

2. Deploy with the env var set (see Cloud Run section below).

## Setup

```bash
# 1. Create .env from example
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (AI Studio)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
python main.py
```

## Deployment (Cloud Run)

```bash
gcloud run deploy accounting-agent \
  --source . \
  --region europe-west1 \
  --project ainm26osl-764 \
  --allow-unauthenticated \
  --timeout=540 \
  --memory=1Gi --cpu=1 \
  --max-instances=3 --concurrency=1 \
  --set-env-vars="GEMINI_API_KEY=<your-ai-studio-key>"
```

## Key Files

| File | Purpose |
|------|---------|
| `agent.py` | Core Gemini agent with tool definitions and handlers |
| `main.py` | FastAPI server, task parsing, pre-fetching |
| `tripletex_client.py` | Tripletex API client |
| `submit_one.py` | Submit 1 task via Chrome CDP |
| `submit_batch.py` | Submit N tasks (currently N=2) |
| `check_scores.py` | Scrape competition scores (`--refresh` to reload) |

## Changelog

### Rev 48 (current)
- Moved `GEMINI_API_KEY` to `.env` (loaded via python-dotenv), removed hardcoded fallback
- Added `.env.example` with placeholder

### Rev 47
- Rate limit retry: upgraded from 5 attempts (2-17s) to 7 attempts with exponential backoff (5-120s) + jitter
- Fixed `amountOutstanding` not valid on SupplierInvoiceDTO — removed from guidance
- Fixed `voucherNumber` not valid on VoucherDTO — added valid fields list to description
- Fixed division creation failing due to missing `organizationNumber` field

### Rev 46
- **Date filter fix**: Auto-bump `dateTo` by +1 day when `dateFrom == dateTo` (Tripletex dateTo is exclusive)
- **Empty response nudge**: After 2 consecutive empty Gemini responses, inject nudge message to unstick model
- **Currency exchange guidance**: Added agio/disagio (valutagevinst/valutatap) instructions for foreign currency invoices
- **Supplier invoice retry**: Exponential backoff (3s, 6s, 12s, 24s) for 500 errors, clear error message after exhaustion
- Updated tool descriptions: dateTo exclusivity notes, isClosed/amountOutstanding warnings

### Rev 43
- Bank reconciliation: Changed from manual vouchers to `pay_supplier_invoice` for supplier payments
- Error correction guidance: Added balance arithmetic rules, VAT handling, efficiency tips

### Rev 42
- Voucher balance pre-flight check: Reject postings that don't sum to 0 before API call
- Repeated error detection: Track last N errors, inject nudge after 3 identical errors
- Fixed hardcoded company ID in division pre-task setup

### Rev 41
- Salary division pre-setup: Auto-create division (virksomhet) before salary tasks
- Enhanced salary auto-fix: dateOfBirth check, division creation, employment linking
- Salary prompt guidance: Full flow documentation

### Rev 38-40
- Model swap: gemini-2.5-pro → gemini-3.1-pro-preview (better reasoning)
- Thinking config: budget=24576, max_output=16384, temp=0.0

### Earlier revisions
- Activity creation fix (activityType=INTEGER)
- Travel expense sequential ordering
- Project fixedprice lowercase
- Error correction supplier on 2400
- Employee onboarding EXTENDED userType
- Payment reversal negative paidAmount
- Double-brace escaping in _safe_json
- Token usage logging
