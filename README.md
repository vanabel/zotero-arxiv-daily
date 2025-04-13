# Zotero arXiv Daily

A Python tool that automatically fetches relevant arXiv papers based on your Zotero library and sends them to your email daily.

## Features

- Fetches new papers from arXiv based on your interests
- Uses your Zotero library to determine relevance
- Generates TLDR summaries using LLMs (OpenAI or local models)
- Sends beautifully formatted daily emails with paper recommendations
- Caches results for better performance
- Supports both OpenAI API and local LLM models
- Configurable arXiv queries and paper limits

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zotero-arxiv-daily.git
cd zotero-arxiv-daily
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and fill in your credentials:
```bash
cp .env.example .env
```

## Configuration

Edit the `.env` file with your credentials and preferences:

- Zotero API credentials (required)
- OpenAI API configuration (optional)
- Local LLM configuration (optional)
- Email settings (required)
- Cache and logging settings
- ArXiv query configuration

### Zotero API Setup

1. Go to [Zotero's API key page](https://www.zotero.org/settings/keys)
2. Create a new API key
3. Note down your User ID and API Key

### Email Setup

For Gmail users:
1. Enable [2-Step Verification](https://myaccount.google.com/security)
2. Generate an [App Password](https://myaccount.google.com/apppasswords)
3. Use the App Password in your `.env` file

## Usage

Run the script with default settings:
```bash
python main.py
```

### Command Line Options

- `--zotero-id`: Your Zotero user ID
- `--zotero-key`: Your Zotero API key
- `--arxiv-query`: Custom arXiv query
- `--max-paper-num`: Maximum number of papers to process
- `--use-llm-api`: Use OpenAI API instead of local LLM
- `--openai-api-key`: OpenAI API key
- `--openai-api-base`: Custom OpenAI API base URL
- `--model-name`: Model name for OpenAI API
- `--language`: Language for TLDR generation (default: en)
- `--debug`: Enable debug mode
- `--no-email-send`: Don't send email (for testing)
- `--send-empty`: Send email even if no papers found

Example:
```bash
python main.py --zotero-id 123456 --zotero-key abcdef --arxiv-query "cat:cs.CV" --max-paper-num 5
```

## How It Works

1. Fetches your Zotero library to understand your interests
2. Retrieves new papers from arXiv based on your query
3. Computes similarity between new papers and your Zotero library
4. Generates TLDR summaries for the most relevant papers
5. Sends a formatted email with recommendations

## Customization

### ArXiv Query

You can customize the arXiv query in the `.env` file or via command line. Example queries:
- `cat:cs.CV` - Computer Vision papers
- `cat:cs.AI` - Artificial Intelligence papers
- `cat:cs.LG` - Machine Learning papers
- `cat:cs.CL` - Computation and Language papers
- `cat:cs.NE` - Neural and Evolutionary Computing papers
- `cat:stat.ML` - Machine Learning papers from Statistics

### Email Template

The email template can be customized in `construct_email.py`. It supports:
- Ordered list of papers
- Paper titles with links
- Author information
- Relevance scores
- TLDR summaries
- PDF and code links

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
