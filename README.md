#  Agent-as-Coder: Bank Statement Parser

This project implements an **autonomous coding agent** that writes, tests, and self-corrects a custom parser for bank statement PDFs.  
The agent reads the provided **ICICI Bank sample statement** and automatically generates a parser (`icici_parser.py`) that produces a DataFrame matching the expected CSV.

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pallavi707/ai-agent-challenge.git
   cd ai-agent-challenge

2. **Install dependencies**
    pip install -r requirements.txt

3. **Create a .env file in the project root directory. Add the api key as shown below:**
    GOOGLE_API_KEY=your_api_key_here

4. **Run the agent**
    python agent.py --target icici

**Project Structure**

```text

ai-agent-challenge/
│
├── agent.py                 # Core agent (generation + testing + self-fix loop)
├── custom_parser/
│   ├── __init__.py
│   └── icici_parser.py      # Auto-generated parser (final version)
│
├── artifacts/               # Debug archive (auto-created during execution)
│   ├── attempt_1.py         # First generation attempt
│   ├── attempt_2.py         # After first self-correction
│   └── attempt_3.py         # After second self-correction
│
├── data/
│   └── icici/
│       ├── icici sample.pdf # Input sample PDF
│       └── result.csv       # Expected parsed CSV output
│
├── requirements.txt         # Dependencies
└── .env                     # Contains GOOGLE_API_KEY

```
```
**Agent Workflow**

generate_parser → run_tests → decide_next 
                                    ↓
                    ┌───────────────┼──────────────┐
                    ↓               ↓              ↓
              "success"        "retry"        "failure"
                    ↓               ↓              ↓
            report_success ← self_correct → run_tests (≤3 attempts)
                    ↓
                   END
```
