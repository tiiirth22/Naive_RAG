import os
import csv
import wikipedia
from pathlib import Path
from config import DATA_PATH


def scrape_wikipedia_pages(keywords_file: str) -> dict:
    """
    Scrape Wikipedia pages based on keywords from CSV file.
    
    Args:
        keywords_file: Path to CSV file with columns: keyword, pages
        
    Returns:
        Dictionary with keyword -> list of scraped articles
    """
    if not os.path.exists(keywords_file):
        raise FileNotFoundError(f"Keywords file not found: {keywords_file}")
    
    scraped_data = {}
    
    with open(keywords_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            keyword = row.get('keyword', '').strip()
            try:
                num_pages = int(row.get('pages', 1))
            except ValueError:
                num_pages = 1
            
            if not keyword:
                continue
            
            print(f"Scraping Wikipedia for keyword: {keyword}")
            
            try:
                # Search for the keyword
                search_results = wikipedia.search(keyword, results=num_pages)
                
                if not search_results:
                    print(f"  No results found for {keyword}")
                    continue
                
                articles = []
                for page_title in search_results:
                    try:
                        page = wikipedia.page(page_title, auto_suggest=True)
                        article_data = {
                            'title': page.title,
                            'url': page.url,
                            'content': page.content
                        }
                        articles.append(article_data)
                        print(f"  ✓ Scraped: {page.title}")
                    except wikipedia.exceptions.DisambiguationError as e:
                        print(f"  ⚠ Disambiguation error for {page_title}: {str(e)[:50]}...")
                    except wikipedia.exceptions.PageError:
                        print(f"  ✗ Page not found: {page_title}")
                    except Exception as e:
                        print(f"  ✗ Error scraping {page_title}: {str(e)}")
                
                scraped_data[keyword] = articles
                
            except Exception as e:
                print(f"  ✗ Error searching for {keyword}: {str(e)}")
    
    return scraped_data


def save_scraped_data(scraped_data: dict) -> None:
    """
    Save scraped Wikipedia data to disk.
    
    Args:
        scraped_data: Dictionary with keyword -> list of articles
    """
    import json
    
    os.makedirs(DATA_PATH, exist_ok=True)
    
    for keyword, articles in scraped_data.items():
        # Create a safe filename from keyword
        safe_keyword = "".join(c for c in keyword if c.isalnum() or c in ('-', '_')).rstrip()
        filename = f"{DATA_PATH}/{safe_keyword}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(articles)} articles for '{keyword}' to {filename}")


def load_scraped_data() -> dict:
    """
    Load all scraped data from disk.
    
    Returns:
        Dictionary with keyword -> list of articles
    """
    import json
    
    scraped_data = {}
    
    if not os.path.exists(DATA_PATH):
        return scraped_data
    
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.json'):
            filepath = os.path.join(DATA_PATH, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                keyword = filename.replace('.json', '')
                scraped_data[keyword] = json.load(f)
    
    return scraped_data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scrape_wikipedia.py <keywords_csv_file>")
        print("Example: python scrape_wikipedia.py keywords.csv")
        sys.exit(1)
    
    keywords_file = sys.argv[1]
    print(f"Loading keywords from {keywords_file}...\n")
    
    scraped_data = scrape_wikipedia_pages(keywords_file)
    save_scraped_data(scraped_data)
    
    print(f"\n✓ Scraping complete! Total keywords: {len(scraped_data)}")
    total_articles = sum(len(articles) for articles in scraped_data.values())
    print(f"✓ Total articles scraped: {total_articles}")