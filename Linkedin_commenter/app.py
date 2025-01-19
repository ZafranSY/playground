import warnings
from typing import Dict, Optional
from flask import Flask, render_template, request
from jamaibase import JamAI, protocol as p

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Flask app
app = Flask(__name__)

class LinkedInCommenter:
    def __init__(self, project_id: str, pat: str):
        self.client = JamAI(
            project_id=project_id,
            token=pat
        )
    
    def validate_input(self, post_text: str, comment_text: str) -> bool:
        """Validate input text."""
        if not post_text or not comment_text:
            raise ValueError("Post text and comment cannot be empty")
        return True
    
    def extract_comment(self, comment_text: str) -> str:
        """Extract the content inside the quotes."""
        # Find the text inside quotes (ignoring the prefix like "Here's a possible roast: ")
        start = comment_text.find('"')
        end = comment_text.rfind('"')
        
        if start != -1 and end != -1 and start < end:
            return comment_text[start + 1:end]  # Extract the content inside the quotes
        return comment_text  # Return the original if no quotes are found
    
    def process_post(self, post_text: str, comment_text: str) -> Optional[Dict[str, str]]:
        """Process LinkedIn post content and generate comment and keywords."""
        try:
            # Validate input post text and comment text
            self.validate_input(post_text, comment_text)
            
            # Add both post_text and comment_text to action table for processing
            response = self.client.table.add_table_rows(
                table_type=p.TableType.action,
                request=p.RowAddRequest(
                    table_id="linkedin_commenter",
                    data=[
                        {"content": post_text},
                        {"comment": comment_text}
                    ],
                    stream=False,
                ),
            )
            
            # Extract results from the table
            comment = response.rows[0].columns["commnet"].text
            # Extract the content inside the quotes for commnet
            comment = self.extract_comment(comment)

            # Extract linkedi_commeter and Hate_commenet
            linkedi_commeter = response.rows[0].columns.get("linkedi_commeter", {}).get("text", "No LinkedIn comment generated")
            Hate_commenet = response.rows[0].columns.get("Hate_commenet", {}).get("text", "No hate comment generated")

            results = {
                "commnet": comment,
                "keywords": response.rows[0].columns["keywords"].text,
                "linkedi_commeter": linkedi_commeter,
                "Hate_commenet": Hate_commenet
            }
            return results
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

# Route to render form and process input
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        post_text = request.form['post_text']
        comment_text = request.form['comment_text']
        
        # Set up project and PAT for the client
        PROJECT_ID = "proj_06bd2b114f77a9290089c6f6"
        PAT = "jamai_pat_7a6480bb4d65eae97477156859ea01286220df2eaf3cb699"
        
        # Initialize LinkedInCommenter instance
        commenter = LinkedInCommenter(PROJECT_ID, PAT)
        
        # Process the post and comment
        result = commenter.process_post(post_text, comment_text)
        
        if result:
            linkedi_commeter=result['linkedi_commeter']
            start = linkedi_commeter.find('"')
            end = linkedi_commeter.rfind('"')
            Hate_commenet=result['Hate_commenet']
            start = Hate_commenet.find('"')
            end = Hate_commenet.rfind('"')
            return render_template('index.html', 
                                   comment=result['commnet'], 
                                   keywords=result['keywords'],
                                   linkedi_commeter = linkedi_commeter,
                                   Hate_commenet=Hate_commenet)
        else:
            return render_template('index.html', error="Failed to generate comment or extract keywords.")
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
