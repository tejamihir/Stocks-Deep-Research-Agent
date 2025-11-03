import os
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
import pdfkit

# 1) Load S&P 500 tickers from local CSV (ignore Wikipedia)
repo_root = Path(__file__).resolve().parents[1]
tickers_csv = repo_root / 'data' / 'yfinance_outputs' / 'sp500_tickers.csv'
df_sp500 = pd.read_csv(tickers_csv)
symbol_col = 'Symbol' if 'Symbol' in df_sp500.columns else df_sp500.columns[0]
tickers = (
    df_sp500[symbol_col]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

# 2) Setup download directories (capitalized Data/Annual Reports)
base_dir = repo_root / 'Data' / 'Annual Reports'
edgar_root = base_dir / 'sec-edgar-filings'
pdf_dir = base_dir / 'pdfs'
base_dir.mkdir(parents=True, exist_ok=True)
pdf_dir.mkdir(parents=True, exist_ok=True)

# 3) Initialize SEC Edgar Downloader (provide contact info) and write under base_dir
dl = Downloader("Mihir Rao", "raomihirs@gmail.com", download_folder=str(base_dir))

def find_10k_file(ticker: str):
    ticker_dir = os.path.join(str(edgar_root), ticker, "10-K")
    if not os.path.exists(ticker_dir):
        return None, None
    years = sorted(os.listdir(ticker_dir), reverse=True)
    for year in years:
        year_dir = os.path.join(ticker_dir, year)
        # Prefer html, then txt, then xml
        for file in os.listdir(year_dir):
            if file.endswith('.htm') or file.endswith('.html'):
                return os.path.join(year_dir, file), 'html'
        for file in os.listdir(year_dir):
            if file.endswith('.txt'):
                return os.path.join(year_dir, file), 'txt'
        for file in os.listdir(year_dir):
            if file.endswith('.xml'):
                return os.path.join(year_dir, file), 'xml'
    return None, None

# 4) Download latest 10-K for each ticker
downloaded: list[dict] = []
os.chdir(base_dir)  # ensure downloads land under Data/Annual Reports
for t in tickers:
    try:
        dl.get('10-K', t, limit=1)
        path, ftype = find_10k_file(t)
        if path:
            downloaded.append({'ticker': t, 'path': path, 'type': ftype})
        else:
            print(f'No valid 10-K found for {t}')
    except Exception as e:
        print(f'Error for {t}: {e}')

# 5) Convert downloaded reports to PDF (HTML or TXT; XML skipped)
for entry in downloaded:
    try:
        ticker = entry['ticker']
        path = entry['path']
        ftype = entry['type']
        pdf_path = pdf_dir / f'{ticker}_latest_10k.pdf'
        if ftype == 'html':
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            texts = soup.prettify()
            pdfkit.from_string(texts, str(pdf_path))
            print(f'Saved PDF for {ticker} -> {pdf_path}')
        elif ftype == 'txt':
            from fpdf import FPDF
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            for line in text.split('\n'):
                # Use multi_cell to wrap long lines
                pdf.multi_cell(0, 5, line)
            pdf.output(str(pdf_path))
            print(f'Saved PDF for {ticker} (from txt) -> {pdf_path}')
        else:
            # XML not processed
            print(f'Skipping XML for {ticker}: {path}')
    except Exception as e:
        print(f'PDF conversion failed for {entry["ticker"]}: {e}')
