# Debugging Yahoo Finance News on Streamlit Cloud

## What Was Added

Comprehensive debug logging has been added to track the Yahoo Finance news flow:

1. **`get_top_news_yahoo`** - Logs RSS feed fetching, entry count, and errors
2. **`summarize_news_with_openai`** - Logs OpenAI client creation and summarization
3. **`get_yahoo_news_section`** - Logs captured output length and preview
4. **`rag_answer`** - Logs when news is fetched and if it's empty

All debug messages are printed to `stderr` (so they appear in Streamlit Cloud logs) and prefixed with `DEBUG:`.

## How to View Debug Logs on Streamlit Cloud

1. **Go to your Streamlit Cloud dashboard**
2. **Click on your app**
3. **Click "Manage app" → "Logs"** (or use the hamburger menu → "View logs")
4. **Search for "DEBUG:"** in the logs to see all debug messages

## What to Look For

### If RSS feed fails:
```
DEBUG: Fetching RSS from https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL...
DEBUG: RSS parse error: [error details]
```
**Solution**: Yahoo RSS might be blocked on Streamlit Cloud. Consider using an alternative news source.

### If no entries found:
```
DEBUG: Found 0 entries from RSS feed
DEBUG: No entries found in RSS feed
```
**Solution**: RSS feed might be empty or Yahoo is rate-limiting. Try a different ticker or add retry logic.

### If OpenAI summarization fails:
```
DEBUG: Failed to get OpenAI client: OPENAI_API_KEY environment variable not set.
```
**Solution**: Make sure `OPENAI_API_KEY` is set in Streamlit Cloud secrets.

### If summarization succeeds but output is empty:
```
DEBUG: LLM summary generated, length=150
DEBUG: Captured output length: 0
```
**Solution**: The `redirect_stdout` might not be capturing prints. Check if prints are going to stdout vs stderr.

## Testing Locally

Run this to see debug output in your terminal:

```bash
cd /Users/mihirrao/mihir/Stocks-Deep-Research-Agent
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.generate_answers_deployment import get_yahoo_news_section
output = get_yahoo_news_section('AAPL')
print(f'Final output: {output[:200]}...')
"
```

## Common Issues

1. **RSS feed blocked**: Streamlit Cloud might block Yahoo RSS. Solution: Use NewsAPI or FMP as fallback.
2. **feedparser not installed**: Check `requirements.txt` includes `feedparser>=6.0.11`
3. **OpenAI API key missing**: Set in Streamlit Cloud secrets
4. **Network timeout**: RSS feed might timeout. Add retry logic or increase timeout.

## Next Steps

After deploying with debug logging:
1. Run a query with a ticker (e.g., "Analyze AAPL")
2. Check Streamlit Cloud logs
3. Look for the DEBUG messages to identify where the flow breaks
4. Share the relevant DEBUG lines for further troubleshooting

