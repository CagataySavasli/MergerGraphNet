import pandas as pd
import re
from sec_cik_mapper import StockMapper  # Explicitly import the CIK lookup function
# from edgar.company import Company
from edgartools import Company
from sec_parsers import Filing, download_sec_filing, set_headers
from sec_downloaders import SEC_Downloader
import sec_parser as sp
from typing import List, Tuple


class DataDownloader:
    def __init__(self, config: dict, output_file: str):
        """Initialize the DataDownloader with required parameters."""
        self.config = config
        self.start_year = self.config['data']['start_year']
        self.end_year = self.config['data']['end_year']
        self.sp500_file = self.config['paths']['sp500_historical']
        self.output_file = output_file

        self.mapper = StockMapper()
        self.get_cik = self.mapper.ticker_to_cik

        self.df_sp500 = pd.read_csv(self.sp500_file)
        self.preprocess_historical_sp500()

        # Initialize a list to accumulate all rows
        self.all_rows = []

        # Set headers for HTTP requests
        self.set_headers(self.config["project"]["developer"], self.config["project"]["email"])

        # Instantiate the SEC downloader and set headers
        self.downloader = SEC_Downloader()
        self.downloader.set_headers(self.config["project"]["developer"], self.config["project"]["email"])

    def preprocess_historical_sp500(self):
        self.df_sp500['year'] = self.df_sp500['date'].apply(lambda x: int(x.split('-')[0]))
        self.df_sp500['tickers'] = self.df_sp500['tickers'].apply(lambda x: x.split(','))
        self.df_sp500.drop(columns=['date'], inplace=True)
        self.df_sp500 = self.df_sp500.groupby('year').agg({'tickers': list}).reset_index()
        self.df_sp500['tickers'] = self.df_sp500['tickers'].apply(lambda x: list(set(sum(x, []))))

    def set_headers(self, name: str, email: str):
        """Sets headers for HTTP requests."""
        set_headers(name, email)

    def clean_text(self, text: str) -> str:
        """Cleans up the text by removing unwanted characters and extra whitespace."""
        text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def get_sp500(self, year: int) -> List[str]:
        """Returns a list of unique tickers for a given year."""
        tmp_df = self.df_sp500.loc[self.df_sp500['year'] == year]
        return list(dict.fromkeys([ticker for sublist in tmp_df["tickers"].to_list() for ticker in sublist]))

    def get_reports(self, ticker: str, year: int):
        """Fetches reports for a given ticker and year."""
        try:
            # Lookup the CIK using the official edgar function
            cik = self.get_cik[ticker]
            if not cik:
                print(f"CIK not found for ticker {ticker}")
                return []
            filings = Company(ticker, cik).get_filings(form=["10-K", "10-Q"])
            reports = filings.filter(date=f"{year}-01-01:{year}-12-31")
            return reports
        except Exception as e:
            print(f"Error fetching reports for {ticker}: {e}")
            return []

    def get_mda_10k(self, html: str) -> str:
        """Extracts MD&A section from a 10-K filing."""
        section = "item 7"
        filing = Filing(html)
        filing.parse()
        item = filing.find_section_from_title(section)
        if item is None:
            raise ValueError("MD&A section not found in 10-K filing")
        return filing.get_text_from_section(item, include_title=True)

    def get_mda_10q(self, html: str) -> str:
        """Extracts MD&A section from a 10-Q filing."""
        elements = sp.Edgar10QParser().parse(html)
        tree = sp.TreeBuilder().build(elements)
        # Gather all top-level sections from the built tree
        top_level_sections = [section for part in tree for section in part.children]
        mdna_sections = [sec for sec in top_level_sections
                         if "management" in sec.semantic_element.text.lower()]
        if len(mdna_sections) != 1:
            raise ValueError("Could not uniquely identify the MD&A section in 10-Q filing")
        mdna_section = mdna_sections[0]

        # Convert section to markdown format based on official usage
        descendants = list(mdna_section.get_descendants())
        levels = sorted({node.semantic_element.level for node in descendants
                         if isinstance(node.semantic_element, sp.TitleElement)})
        level_to_markdown = {level: "#" * (i + 2) for i, level in enumerate(levels)}

        markdown = f"# {mdna_section.semantic_element.text}\n"
        for node in descendants:
            element = node.semantic_element
            if isinstance(element, sp.TextElement):
                markdown += f"{element.text}\n"
            elif isinstance(element, sp.TitleElement):
                markdown += f"{level_to_markdown.get(element.level, '##')} {element.text}\n"
            elif isinstance(element, sp.TableElement):
                markdown += f"[{element.get_summary()}]\n"
        return markdown

    def get_mda(self, report) -> Tuple[str, str]:
        """Extracts the MD&A section from a given report."""
        try:
            form = report.form
            # Construct the URL based on official usage (remove hyphens and index.html)
            url = report.homepage_url.replace("-", "").replace("index.html", "/" + report.primary_document)
            html = download_sec_filing(url)

            if form == "10-K":
                item_text = self.get_mda_10k(html)
            elif form == "10-Q":
                item_text = self.get_mda_10q(html)
            else:
                print(f"Unsupported form: {form}")
                return "error", "error"

            return self.clean_text(item_text), form
        except Exception as e:
            print(f"Error extracting MD&A for report: {e}")
            return "error", "error"

    def download_reports(self):
        """Main method to download reports for the defined range of years."""
        for year in range(self.start_year, self.end_year + 1):
            print(f"Processing reports for {year}")
            tickers = self.get_sp500(year)
            year_rows = []  # Accumulate rows for this year
            for idx, ticker in enumerate(tickers, start=1):
                print(f"Processing {idx}/{len(tickers)} for {ticker}")
                reports = self.get_reports(ticker, year)
                if not reports:
                    continue

                for report in reports:
                    mda, form = self.get_mda(report)
                    if mda == "error":
                        continue

                    row = {
                        "ticker": ticker,
                        "form": form,
                        "year": year,
                        "filing_date": report.filing_date,
                        "mda": mda
                    }
                    year_rows.append(row)
                    self.all_rows.append(row)

            # Create a DataFrame for the year and save to CSV
            if year_rows:
                year_df = pd.DataFrame(year_rows)
                year_df.to_csv(f"./data/raw/years/reports_{year}.csv", index=False)
                print(f"Year {year} completed with {len(year_df)} reports.")
            else:
                print(f"Year {year} completed with 0 reports.")

        # Save the combined results for all years
        if self.all_rows:
            all_df = pd.DataFrame(self.all_rows)
            all_df.to_csv(self.output_file, index=False)
            print(f"All reports downloaded and saved to {self.output_file}.")
        else:
            print("No reports downloaded.")
