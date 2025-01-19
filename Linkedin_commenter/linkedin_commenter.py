import warnings
import os
from typing import Dict, Optional
from jamaibase import JamAI, protocol as p

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

class LinkedInCommenter:
    def __init__(self, project_id: str, pat: str):
        self.client = JamAI(
            project_id=project_id,
            token=pat
        )
    
    def validate_input(self, post_text: str,comment_text:str) -> bool:
        """Validate input text."""
        if not post_text or not comment_text:
            raise ValueError("Post text cannot be empty")
        return True
    
    def process_post(self, post_text: str, comment_text:str) -> Optional[Dict[str, str]]:
        """Process LinkedIn post content and generate comment and keywords."""
        try:
            # Validate input post text
            self.validate_input(post_text,comment_text)
            
            # Add post text to action table for processing
            response = self.client.table.add_table_rows(
                table_type=p.TableType.action,
                request=p.RowAddRequest(
                    table_id="linkedin_commenter",
 data=[
                        {"content": post_text},
                        {"comment": comment_text}
                    ],                    stream=False,
                ),
                
            )
            
            # Extract results from the table
            results = {
                "commnet": response.rows[0].columns["commnet"].text,
                "keywords": response.rows[0].columns["keywords"].text
            }
            return results
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

def main():
    # Set up project and PAT for the client
    PROJECT_ID = "proj_06bd2b114f77a9290089c6f6"
    PAT = "jamai_pat_7a6480bb4d65eae97477156859ea01286220df2eaf3cb699"
    
    # Initialize LinkedInCommenter instance
    commenter = LinkedInCommenter(PROJECT_ID, PAT)
    
    # User input for LinkedIn post text
    post_text = input("Enter LinkedIn post text: ")
    comment_text = input("Enter comment sample : ")
    # Process the post
    result = commenter.process_post(post_text,comment_text)
    
    if result:
        print("\nGenerated Comment:")
        print(result['commnet'])
        print("\nExtracted Keywords:")
        print(result['keywords'])
    else:
        print("Failed to generate comment or extract keywords.")

if __name__ == "__main__":
    main()
