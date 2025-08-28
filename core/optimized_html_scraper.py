#!/usr/bin/env python3
"""
Optimized EuroMillions HTML Scraper
Efficiently extracts draw results from resultados_euromilhoes.html
"""

import re
from datetime import datetime
import csv

class OptimizedEuroMillionsHTMLScraper:
    def __init__(self, html_file_path):
        self.html_file_path = html_file_path
        self.results = []
    
    def parse_date(self, date_string):
        """Parse date from format 'Tuesday - 19/08/2025'"""
        try:
            date_part = date_string.split(' - ')[-1].strip()
            return datetime.strptime(date_part, '%d/%m/%Y')
        except Exception:
            return None
    
    def extract_draw_data(self):
        """Extract draw data using regex patterns for better performance"""
        print("Processing HTML file with regex patterns...")
        
        with open(self.html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Pattern to find draw containers with dates and IDs
        date_pattern = r'<a[^>]*data-draw-id="([^"]+)"[^>]*>.*?<div class="c-resultado-sorteo__cabecera--euromillones">.*?([A-Za-z]+ - \d{2}/\d{2}/\d{4})</p>'
        
        # Find all date matches
        date_matches = re.findall(date_pattern, html_content, re.DOTALL)
        print(f"Found {len(date_matches)} potential draws")
        
        for draw_id, date_string in date_matches:
            draw_date = self.parse_date(date_string)
            if not draw_date:
                continue
            
            # Extract main numbers for this draw_id
            main_numbers_pattern = f'<ul id="qa_resultado-combination-actMainNumbers-EMIL-{re.escape(draw_id)}"[^>]*>(.*?)</ul>'
            main_match = re.search(main_numbers_pattern, html_content, re.DOTALL)
            
            if not main_match:
                continue
                
            # Extract numbers from <li> tags
            number_pattern = r'<li[^>]*>(\d+)</li>'
            main_numbers = [int(n) for n in re.findall(number_pattern, main_match.group(1))]
            
            # Extract star numbers for this draw_id
            stars_pattern = f'<ul id="qa_resultado-combination-actMainNumbers-EMIL-stars-{re.escape(draw_id)}"[^>]*>(.*?)</ul>'
            stars_match = re.search(stars_pattern, html_content, re.DOTALL)
            
            star_numbers = []
            if stars_match:
                star_numbers = [int(n) for n in re.findall(number_pattern, stars_match.group(1))]
            
            # Validate draw (5 main + 2 stars)
            if len(main_numbers) == 5 and len(star_numbers) == 2:
                self.results.append({
                    'date': draw_date,
                    'main_numbers': main_numbers,  # Keep original order
                    'star_numbers': star_numbers,  # Keep original order
                    'draw_id': draw_id
                })
                print(f"✓ {draw_date.strftime('%d/%m/%Y')}: {main_numbers} + {star_numbers}")
            else:
                print(f"✗ Invalid draw {draw_date.strftime('%d/%m/%Y')}: {len(main_numbers)} main, {len(star_numbers)} stars")
        
        # Sort by date (oldest first)
        self.results.sort(key=lambda x: x['date'])
        print(f"\nTotal valid draws: {len(self.results)}")
        return self.results
    
    def save_to_csv(self, output_file='scraped_euromillions_results.csv'):
        """Save results to CSV in chronological order"""
        if not self.results:
            print("No results to save")
            return
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Date', 'Main1', 'Main2', 'Main3', 'Main4', 'Main5', 'Star1', 'Star2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'Date': result['date'].strftime('%Y-%m-%d'),
                    'Main1': result['main_numbers'][0],
                    'Main2': result['main_numbers'][1],
                    'Main3': result['main_numbers'][2],
                    'Main4': result['main_numbers'][3],
                    'Main5': result['main_numbers'][4],
                    'Star1': result['star_numbers'][0],
                    'Star2': result['star_numbers'][1]
                })
        
        print(f"Results saved to {output_file}")
    
    def show_summary(self):
        """Display extraction summary"""
        if not self.results:
            print("No results extracted")
            return
        
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Total draws: {len(self.results)}")
        print(f"Date range: {self.results[0]['date'].strftime('%d/%m/%Y')} to {self.results[-1]['date'].strftime('%d/%m/%Y')}")
        
        print(f"\nFirst 5 draws (chronological order):")
        for i in range(min(5, len(self.results))):
            result = self.results[i]
            main = result['main_numbers']
            stars = result['star_numbers']
            print(f"{i+1:2d}. {result['date'].strftime('%d/%m/%Y')}: {main[0]:02d}-{main[1]:02d}-{main[2]:02d}-{main[3]:02d}-{main[4]:02d} + {stars[0]:02d}-{stars[1]:02d}")
        
        print(f"\nLast 5 draws:")
        start_idx = max(0, len(self.results) - 5)
        for i in range(start_idx, len(self.results)):
            result = self.results[i]
            main = result['main_numbers']
            stars = result['star_numbers']
            print(f"{i+1:2d}. {result['date'].strftime('%d/%m/%Y')}: {main[0]:02d}-{main[1]:02d}-{main[2]:02d}-{main[3]:02d}-{main[4]:02d} + {stars[0]:02d}-{stars[1]:02d}")

def main():
    import os
    
    html_file = 'resultados_euromilhoes.html'
    
    if not os.path.exists(html_file):
        print(f"Error: {html_file} not found")
        return 1
    
    print("Optimized EuroMillions HTML Scraper")
    print("=" * 35)
    
    scraper = OptimizedEuroMillionsHTMLScraper(html_file)
    
    try:
        # Extract data
        results = scraper.extract_draw_data()
        
        if results:
            # Show summary
            scraper.show_summary()
            
            # Save to CSV
            scraper.save_to_csv()
            print(f"\n✅ Scraping completed successfully!")
            return 0
        else:
            print("❌ No valid draws found")
            return 1
            
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return 1

if __name__ == "__main__":
    exit(main())