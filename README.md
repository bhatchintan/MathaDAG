# MathaDAG - Mathematics Paper Dependency Graph

A web application that creates a directed acyclic graph (DAG) of mathematics papers based on their actual dependencies, using the Semantic Scholar API and Google's Gemini AI for intelligent analysis.

## Key Features

- **Intelligent Dependency Detection**: Uses Gemini 2.5 Pro AI to identify true mathematical dependencies (theorems, lemmas, definitions used in proofs)
- **Visual Graph Representation**: Interactive visualization showing dependency relationships
- **Detailed Reasoning**: Shows why each paper is considered a dependency

## Installation

1. Install Python 3 if not already installed

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the Gemini API key set in `app.py`

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and go to: http://127.0.0.1:5000

3. Enter a paper identifier, click Analyze

## Troubleshooting

### "Paper not found" error
- Check that the DOI/ID is correct
- Try different formats (with/without "arXiv:" prefix)
- Some papers may not be in Semantic Scholar's database

### Rate limiting errors
- The app includes automatic retry logic with delays
- If persistent, wait a few minutes before trying again
- Consider getting a Semantic Scholar API key for higher limits

### Gemini API errors
- The app uses Gemini 2.5 Pro for best accuracy
- If overloaded, it will automatically retry
- Check that the API key is valid

## File Structure

- `app.py` - Main Flask application
- `paper_content_fetcher.py` - Handles PDF downloading and text extraction
- `index.html` - Frontend interface
- `requirements.txt` - Python dependencies
- `paper_cache/` - Directory for cached paper content (created automatically)

## Known Errors

- Gemini can classify some citations incorrectly (however, verification can be done by clicking the edge)
- If full paper content is behind a paywall, it can't be accessed for intelligent dependency extraction