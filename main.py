from dotenv import load_dotenv
import ollama
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

load_dotenv()

def normalize_url(url: str) -> str:
    """Ensure URL has a scheme. If missing, prepend https://"""
    parsed = urlparse(url)
    if not parsed.scheme:  # no http/https given
        url = "https://" + url
    return url

def is_url_valid(url: str) -> bool:
    parsed = urlparse(url)
    is_valid = all([parsed.scheme, parsed.netloc])
    if not is_valid:
        print("Invalid URL. Please enter a valid URL (e.g. example.com)")
        return False
    return True

def is_url_reachable(url: str) -> bool:
    """Check if a URL is reachable (HEAD first, then GET if necessary)."""
    try:
        # Try HEAD request first
        response = requests.head(url, timeout=3, allow_redirects=True)
        if response.status_code >= 400:
            # Some servers reject HEAD, fallback to GET
            response = requests.get(url, timeout=3, stream=True, allow_redirects=True)
        if response.status_code >= 400:
            print("Webpage in the given URL is not reachable. Retry with a different URL.")
            return False
        return True
    except requests.RequestException as e:
        print(f"Error reaching the URL: {e}. Retry with a different URL.")
        return False

        
def prompt_for_url():
    while True:
        url = input("Enter the URL of the page to summarize: ").strip().lower()
        url = normalize_url(url)
        if not is_url_valid(url):
            continue
        if not is_url_reachable(url):
            continue
        print(f"Selected URL: {url}")
        return url


def prompt_for_model():
    while True:
        model = input("Enter the model name (Default: gpt-oss): ").strip().lower()
        if not model:
            model = 'gpt-oss'
        print(f"Selected model: {model}")
        return model

def summarize(url: str, model: str):
    text = get_webpage_text(url)
    system_prompt = ("You are an assistant that analyzes the text and provides a short summary, "
                    "ignoring text that might be navigation related. Respond in markdown.")
    user_prompt = (f"You should provide a short summary of this content in markdown. "
                    f"If it includes news or announcements, then summarize these too. "
                    f"The contents to be summarized given as below. {text}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    print(f"Please wait while {model} summarizes the webpage. This might take a while...")
    response = ollama.chat(model=model, messages=messages)
    print(f"\n\n{response['message']['content']}")
    print(f"Summarized the webpage {url}.")

def get_webpage_text(url: str) -> str:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title if soup.title else "Title not found"
        irrelevant_tags = ["script", "style", "img", "input"]
        for tag in soup.body(irrelevant_tags):
            tag.decompose()
        text = soup.body.get_text(strip=True)
        return text
    except requests.RequestException as e:
        raise OSError(f"Failed to retrieve webpage content: {e}")

def main():
    while True:
        try:
            url = prompt_for_url()
            model = prompt_for_model()
            summarize(url, model)
            break
        except (ValueError, OSError) as e:
            print(f"Summarization failed: {e}")
            retry = input("Do you want to retry? (Y/n): ").strip().lower() or 'y'
            if retry != 'y':
                break
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break


if __name__ == '__main__':
    main()
