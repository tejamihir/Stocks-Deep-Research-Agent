import os
from typing import List, Optional

import requests
import yfinance as yf

try:
    import feedparser  # type: ignore
except ImportError:
    feedparser = None

try:
    import openai  # type: ignore
except ImportError:
    openai = None


_openai_client = None


def get_openai_client():
    global _openai_client

    if openai is None:
        raise RuntimeError("openai package not installed; install the 'openai' dependency to enable summaries.")

    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        _openai_client = openai.OpenAI(api_key=api_key)

    return _openai_client


def get_analyst_estimates_and_price_target(ticker: str) -> None:
    stock = yf.Ticker(ticker)

    try:
        targets = stock.analysis
    except AttributeError:
        try:
            targets = stock.get_analysis()
        except AttributeError:
            targets = None
    except Exception:
        targets = None

    try:
        recommendations = stock.recommendations
    except AttributeError:
        recommendations = None

    try:
        fast_info = stock.fast_info
    except AttributeError:
        fast_info = {}

    print(f"Analyst Targets & Estimates for {ticker}:")
    if targets is not None and not targets.empty:
        print(targets)
    else:
        print("No analyst target summary available.")

    if recommendations is not None and not recommendations.empty:
        current_period = recommendations.iloc[0]
        period_metrics = current_period[["strongBuy", "buy", "hold", "sell", "strongSell"]].astype(float)
        top_category = period_metrics.idxmax()
        top_count = period_metrics.max()

        print("\nRecent Analyst Recommendations:")
        print(recommendations.tail(10))
        print(
            f"\n0m period highest analyst count: {top_category} ({top_count}) "
            "(strongBuy > buy > hold > sell > strongSell)."
        )
    else:
        print("No analyst recommendations available.")

    print("\nQuick Price Target Info (if available):")
    for key in ["targetMeanPrice", "targetHighPrice", "targetLowPrice", "targetMedianPrice"]:
        value = fast_info.get(key) if hasattr(fast_info, "get") else None
        if value is not None:
            print(f"{key}: {value}")


def get_top_news_fmp(ticker: str, api_key: str, count: int = 5) -> None:
    if not api_key or api_key == "your_fmp_api_key":
        print("FMP_API_KEY is missing. Set the environment variable to fetch news.")
        return

    url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit={count}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        print(f"Failed to fetch news: {exc}")
        return

    if response.status_code != 200:
        if response.status_code == 403:
            print("Failed to fetch news: 403 (Forbidden). Verify your FMP API key and subscription level.")
        else:
            print("Failed to fetch news:", response.status_code)
        return

    try:
        news_items = response.json()
    except ValueError:
        print("Failed to parse news response.")
        return

    if not news_items:
        print("No news articles available.")
        return

    # print(f"\nTop {min(count, len(news_items))} News Articles for {ticker}:")
    # for i, article in enumerate(news_items[:count], start=1):
    #     title = article.get("title", "Untitled")
    #     published = article.get("publishedDate", "Unknown")
    #     source = article.get("site", "Unknown")
    #     summary = article.get("text", "No summary available.")
    #     article_url = article.get("url", "No URL provided.")

    #     print(f"{i}. {title}")
    #     print(f"   Published: {published}")
    #     print(f"   Source: {source}")
    #     print(f"   Summary: {summary}")
    #     print(f"   URL: {article_url}\n")


def fetch_news_newsapi(
    query: str = "uranium industry",
    page_size: int = 10,
    summarize: bool = False,
    summarize_outlook: bool = False,
) -> None:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("NEWSAPI_KEY environment variable is missing. Set it to fetch NewsAPI articles.")
        return

    url = (
        "https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"pageSize={page_size}&"
        f"sortBy=publishedAt&"
        f"apiKey={api_key}"
    )

    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        print(f"Failed to fetch NewsAPI articles: {exc}")
        return

    if response.status_code != 200:
        print("Failed to fetch NewsAPI articles:", response.status_code)
        return

    try:
        data = response.json()
    except ValueError:
        print("Failed to parse NewsAPI response.")
        return

    articles = data.get("articles", [])
    if not articles:
        print("No NewsAPI articles available.")
        return

    #print(f"\nTop {min(page_size, len(articles))} NewsAPI Articles for query '{query}':")
    summarizable_articles: List[dict] = []
    for i, article in enumerate(articles[:page_size], start=1):
        title = article.get("title") or "Untitled"
        published_at = article.get("publishedAt") or "Unknown"
        source_name = article.get("source", {}).get("name") or "Unknown"
        url = article.get("url") or "No URL provided."
        description = article.get("description") or article.get("content") or ""

        # print(f"{i}. {title}")
        # print(f"   Published: {published_at}")
        # print(f"   Source: {source_name}")
        # print(f"   URL: {url}\n")

        summarizable_articles.append(
            {
                "title": title,
                "link": url,
                "published": published_at,
                "summary": description,
            }
        )

    if summarize_outlook:
        summarize_industry_outlook_with_openai(query, summarizable_articles)
    elif summarize:
        summarize_news_with_openai(query, summarizable_articles)


def summarize_news_with_openai(ticker: str, articles: List[dict], model: Optional[str] = None) -> None:
    if not articles:
        return

    try:
        client = get_openai_client()
    except RuntimeError as exc:
        print(f"Skipping OpenAI summarization: {exc}")
        return

    selected_model = model or os.getenv("OPENAI_NEWS_MODEL", "gpt-4o-mini")

    formatted_articles = []
    for idx, article in enumerate(articles, start=1):
        lines = [f"Title: {article.get('title', 'Untitled')}"]
        if article.get("published"):
            lines.append(f"Published: {article['published']}")
        lines.append(f"Link: {article.get('link', 'No link available.')}")
        if article.get("summary"):
            lines.append(f"Snippet: {article['summary']}")
        formatted_articles.append(f"Article {idx}:\n" + "\n".join(lines))

    prompt = (
        f"You are a financial research assistant. Summarize the key takeaways from recent Yahoo Finance news for {ticker}. "
        "Use the information below to produce exactly five concise bullet points capturing the most important themes, "
        "developments, or risks. If there is insufficient information, explicitly state that."
    )

    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "You create crisp financial news briefs."},
                {"role": "user", "content": prompt + "\n\n" + "\n\n".join(formatted_articles)},
            ],
            temperature=0.2,
            max_tokens=500,
        )
    except Exception as exc:
        print(f"OpenAI summarization failed: {exc}")
        return

    summary_text = response.choices[0].message.content.strip()
    print("\nLLM 5-Point Summary:")
    print(summary_text)


def summarize_industry_outlook_with_openai(industry: str, articles: List[dict], model: Optional[str] = None) -> None:
    if not articles:
        return

    try:
        client = get_openai_client()
    except RuntimeError as exc:
        print(f"Skipping OpenAI outlook summarization: {exc}")
        return

    selected_model = model or os.getenv("OPENAI_NEWS_MODEL", "gpt-4o-mini")

    formatted_articles = []
    for idx, article in enumerate(articles, start=1):
        lines = [f"Title: {article.get('title', 'Untitled')}"]
        if article.get("published"):
            lines.append(f"Published: {article['published']}")
        lines.append(f"Link: {article.get('link', 'No link available.')}")
        if article.get("summary"):
            lines.append(f"Snippet: {article['summary']}")
        formatted_articles.append(f"Article {idx}:\n" + "\n".join(lines))

    prompt = (
        f"You are an industry research assistant. Evaluate the future outlook for the {industry} industry based on the articles below. "
        "Highlight expected growth rates (including CAGR or TAM projections), demand drivers, capital expenditure plans, regulatory or technological shifts, "
        "and major risks or constraints that could shape the industry's trajectory. Produce exactly six bullet points focused on forward-looking insights. "
        "If the sources lack clear outlook information, just summarize the information in the articles."
    )

    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "You analyze industries with an emphasis on future outlook and growth dynamics."},
                {"role": "user", "content": prompt + "\n\n" + "\n\n".join(formatted_articles)},
            ],
            temperature=0.2,
            max_tokens=500,
        )
    except Exception as exc:
        print(f"OpenAI outlook summarization failed: {exc}")
        return

    summary_text = response.choices[0].message.content.strip()
    print("\nLLM Industry Outlook Summary:")
    print(summary_text)


def get_top_news_yahoo(ticker: str, count: int = 5, summarize: bool = True) -> None:
    if feedparser is None:
        print("feedparser library not installed; skipping Yahoo Finance RSS news.")
        return

    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(rss_url)

    if hasattr(feed, "bozo_exception") and feed.bozo_exception:
        print(f"Failed to parse Yahoo Finance RSS feed: {feed.bozo_exception}")
        return

    entries = getattr(feed, "entries", [])[:count]
    if not entries:
        print("No Yahoo Finance news articles available.")
        return

    articles: List[dict] = []

    # print(f"\nTop {len(entries)} Yahoo Finance News Articles for {ticker}:")
    for i, entry in enumerate(entries, start=1):
        title = getattr(entry, "title", "Untitled")
        link = getattr(entry, "link", "No link provided.")
        published = getattr(entry, "published", None)
        summary = getattr(entry, "summary", getattr(entry, "description", ""))

    #     print(f"{i}. {title}")
    #     if published:
    #         print(f"   Published: {published}")
    #     print(f"   Link: {link}\n")

        articles.append(
            {
                "title": title,
                "link": link,
                "published": published,
                "summary": summary,
            }
        )

    if summarize:
        summarize_news_with_openai(ticker, articles)


if __name__ == "__main__":
    ticker = "NVDA"
    fmp_api_key = os.getenv("FMP_API_KEY", "your_fmp_api_key")

    get_analyst_estimates_and_price_target(ticker)
    get_top_news_fmp(ticker, fmp_api_key)
    get_top_news_yahoo(ticker)
    fetch_news_newsapi(query="Uranium Industry in the next 10 years", summarize_outlook=True)

