#!/bin/bash
# Debug commands for Yahoo Finance news issue on Streamlit Cloud

echo "=== 1. Test RSS Feed Directly ==="
curl -s "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US" | head -20

echo -e "\n=== 2. Test Python feedparser ==="
python3 -c "
import feedparser
rss_url = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US'
feed = feedparser.parse(rss_url)
print(f'Feed entries: {len(feed.entries)}')
if feed.entries:
    print(f'First entry title: {feed.entries[0].title}')
if hasattr(feed, 'bozo_exception') and feed.bozo_exception:
    print(f'RSS Parse Error: {feed.bozo_exception}')
"

echo -e "\n=== 3. Test get_top_news_yahoo function (without summarization) ==="
cd /Users/mihirrao/mihir/Stocks-Deep-Research-Agent
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.get_ANALYT_ESTIMATES import get_top_news_yahoo
get_top_news_yahoo('AAPL', summarize=False)
"

echo -e "\n=== 4. Test get_top_news_yahoo function (with summarization) ==="
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.get_ANALYT_ESTIMATES import get_top_news_yahoo
get_top_news_yahoo('AAPL', summarize=True)
"

echo -e "\n=== 5. Check Environment Variables ==="
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}..."
echo "NEWSAPI_KEY: ${NEWSAPI_KEY:0:20}..."

echo -e "\n=== 6. Test get_yahoo_news_section from deployment ==="
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.generate_answers_deployment import get_yahoo_news_section
output = get_yahoo_news_section('AAPL')
print(f'Output length: {len(output)}')
print(f'Output preview: {output[:200]}...')
"

