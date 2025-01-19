import requests
import json

# Constants
BASE_URL = "https://api.jamai.com/v1/projects/proj_06bd2b114f77a9290089c6f6"
TABLE_NAME = "linkedin_commenter"
PAT = "jamai_pat_7a6480bb4d65eae97477156859ea01286220df2eaf3cb699"

# Headers
HEADERS = {
    "Authorization": f"Bearer {PAT}",
    "Content-Type": "application/json"
}

def get_table_data():
    """Fetch the table data from the Jamai project."""
    url = f"{BASE_URL}/tables/{TABLE_NAME}/rows"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def post_input_comment(post_text):
    """Generate a comment for the given post text."""
    # Sample logic to simulate comment generation
    return f"Here's a LinkedIn comment: 'Great insights on {post_text}'"

def extract_keywords(text):
    """Extract keywords from the post text. Here we're simulating extraction."""
    return ["Great", "insights", "LinkedIn", "business"]

def display_table_data():
    """Display the data from the 'linkedin_commenter' table."""
    data = get_table_data()
    for row in data.get('rows', []):
        print(f"Post: {row['content']}")
        print(f"Keywords: {row['keywords']}")
        print(f"Comment: {row['comment']}")
        print("-" * 50)

def main():
    """Main function to interact with the user."""
    print("Welcome to the LinkedIn Commenter Application")
    
    # Display data from the table
    display_table_data()
    
    # User input for post text
    post_text = input("Enter post text: ")
    
    # Generate comment based on user input
    comment = post_input_comment(post_text)
    
    # Extract keywords from the post text
    keywords = extract_keywords(post_text)
    
    print("\nGenerated Comment:")
    print(comment)
    
    print("\nExtracted Keywords:")
    print(", ".join(keywords))
    
    # Add new entry to the table
    data = {
        "content": post_text,
        "comment": comment,
        "keywords": keywords
    }
    response = requests.post(f"{BASE_URL}/tables/{TABLE_NAME}/rows", headers=HEADERS, json=data)
    if response.status_code == 200:
        print("\nSuccessfully added to the table.")
    else:
        print(f"\nFailed to add data: {response.status_code}, {response.text}")

if __name__ == "__main__":
    main()
