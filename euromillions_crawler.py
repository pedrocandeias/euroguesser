#!/usr/bin/env python3
"""
EuroMillions Historical Results Crawler

This script crawls the Euro-Millions.com website to collect all historical
lottery results from 2004 to present day and saves them to a CSV file.

Usage: python3 euromillions_crawler.py
"""

import requests
from bs4 import BeautifulSoup
import csv
import re
import time
from datetime import datetime
import sys


class EuroMillionsCrawler:
    def __init__(self):
        self.base_url = "https://www.euro-millions.com/results-history-{}"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.results = []

    def fetch_page(self, year):
        """Fetch the results page for a specific year."""
        url = self.base_url.format(year)
        try:
            print(f"Fetching results for year {year}...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching year {year}: {e}")
            return None

    def parse_date(self, date_text):
        """Parse date from text like 'Tuesday 31st December 2024'."""
        try:
            # Clean up whitespace and newlines
            date_clean = re.sub(r'\s+', ' ', date_text.strip())
            
            # Remove day of week if present
            date_clean = re.sub(r'^[A-Za-z]+\s+', '', date_clean)
            
            # Remove ordinal suffixes (st, nd, rd, th)
            date_clean = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_clean)
            
            # Parse the date
            parsed_date = datetime.strptime(date_clean, "%d %B %Y")
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError as e:
            print(f"Date parsing error for '{date_text}': {e}")
            return None

    def extract_numbers_from_text(self, text):
        """Extract numbers from text content."""
        # Find all numbers in the text
        numbers = re.findall(r'\b\d+\b', text)
        return [int(n) for n in numbers]

    def parse_results_page(self, html_content, year):
        """Parse the HTML content to extract lottery results."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all result entries - look for links that contain date patterns
        result_links = soup.find_all('a', href=re.compile(r'/results/\d{2}-\d{2}-\d{4}'))
        
        for link in result_links:
            try:
                # Extract date from the link text
                date_text = link.get_text().strip()
                
                # Skip non-date links
                if not any(day in date_text for day in ['Tuesday', 'Friday', 'Monday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday']):
                    continue
                
                draw_date = self.parse_date(date_text)
                if not draw_date:
                    continue

                # Find the table row that contains this link
                table_row = link.find_parent('tr')
                if not table_row:
                    continue

                # Find all list items in this table row (these contain the numbers)
                list_items = table_row.find_all('li')
                all_numbers = []
                
                for item in list_items:
                    number_text = item.get_text().strip()
                    if number_text.isdigit():
                        number = int(number_text)
                        if 1 <= number <= 50 or 1 <= number <= 12:  # Valid lottery number
                            all_numbers.append(number)

                # EuroMillions should have exactly 7 numbers (5 main + 2 stars)
                if len(all_numbers) == 7:
                    # First 5 are main numbers, last 2 are stars
                    main_numbers = sorted(all_numbers[:5])
                    star_numbers = sorted(all_numbers[5:7])
                    
                    # Validate ranges
                    if (all(1 <= n <= 50 for n in main_numbers) and 
                        all(1 <= n <= 12 for n in star_numbers)):
                        
                        result = {
                            'date': draw_date,
                            'main_numbers': main_numbers,
                            'star_numbers': star_numbers
                        }
                        
                        self.results.append(result)
                        print(f"  Found: {draw_date} - {main_numbers} + {star_numbers}")
                
            except Exception as e:
                print(f"Error parsing result: {e}")
                continue

    def crawl_all_years(self, start_year=2004, end_year=None):
        """Crawl all years from start_year to end_year (current year if None)."""
        if end_year is None:
            end_year = datetime.now().year

        for year in range(start_year, end_year + 1):
            html_content = self.fetch_page(year)
            if html_content:
                self.parse_results_page(html_content, year)
                # Be respectful to the server
                time.sleep(1)

    def save_to_csv(self, filename="euromillions_historical_results.csv"):
        """Save all results to a CSV file."""
        if not self.results:
            print("No results to save.")
            return

        # Sort results by date
        self.results.sort(key=lambda x: x['date'])

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Date', 'Main1', 'Main2', 'Main3', 'Main4', 'Main5', 'Star1', 'Star2'])
            
            # Write data
            for result in self.results:
                row = [result['date']] + result['main_numbers'] + result['star_numbers']
                writer.writerow(row)

        print(f"Saved {len(self.results)} results to {filename}")

    def save_to_training_format(self, filename="euromillions_training_data.txt"):
        """Save results in the existing training data format (CSV without headers)."""
        if not self.results:
            print("No results to save.")
            return

        # Sort results by date
        self.results.sort(key=lambda x: x['date'])

        with open(filename, 'w') as f:
            for result in self.results:
                # Format: main1,main2,main3,main4,main5,star1,star2
                line = ','.join(map(str, result['main_numbers'] + result['star_numbers']))
                f.write(line + '\n')

        print(f"Saved {len(self.results)} results to {filename} in training format")


def main():
    """Main function to run the crawler."""
    print("EuroMillions Historical Results Crawler")
    print("=" * 40)
    
    crawler = EuroMillionsCrawler()
    
    try:
        # Crawl all years from 2004 to present
        crawler.crawl_all_years(2004)
        
        # Save results in both formats
        crawler.save_to_csv()
        crawler.save_to_training_format()
        
        print(f"\nCrawling completed successfully!")
        print(f"Total results collected: {len(crawler.results)}")
        
    except KeyboardInterrupt:
        print("\nCrawling interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()