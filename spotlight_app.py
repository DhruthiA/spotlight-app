import json
import textwrap
import streamlit as st
from openai import OpenAI
from datetime import datetime
import hashlib
import io
import csv
import os
import pandas as pd
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import PDF libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    try:
        import pypdf
        PDF_SUPPORT = True
        PyPDF2 = pypdf  # Use pypdf as PyPDF2
    except ImportError:
        PDF_SUPPORT = False

# ---------------------------------
# 🔑 API KEY Configuration
# ---------------------------------
# Priority: 1. Streamlit Secrets (for Streamlit Cloud), 2. Environment variable/.env file (for local)
API_KEY = None
MODEL_NAME = "gpt-4.1-mini"

# Try Streamlit secrets first (for Streamlit Cloud deployment)
try:
    API_KEY = st.secrets.get("OPENAI_API_KEY", None)
    if API_KEY:
        MODEL_NAME = st.secrets.get("OPENAI_MODEL_NAME", MODEL_NAME)
except (AttributeError, FileNotFoundError, KeyError):
    # Not using Streamlit secrets (local development), try environment variable or .env file
    # load_dotenv() was called earlier, so .env file values are loaded into os.getenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", MODEL_NAME)

if not API_KEY:
    st.error("❌ **API Key Not Found** - Please configure your API key:\n\n"
             "**For Local Development:**\n"
             "- Create a `.env` file in the project root with: `OPENAI_API_KEY=your-api-key-here`\n\n"
             "**For Streamlit Cloud Deployment:**\n"
             "- Go to your app settings → Secrets\n"
             "- Add: `OPENAI_API_KEY = 'your-api-key-here'`\n\n"
             "The app cannot function without an API key.")
    st.stop()

# -----------------------------
# CSV Database Configuration
# -----------------------------
# Use absolute path to ensure persistence across sessions
CSV_DB_FILE = os.path.join(os.getcwd(), "story_database.csv")

# -----------------------------
# Helper Functions
# -----------------------------
def get_content_hash(text: str) -> str:
    """Generate SHA256 hash of content for caching."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def retry_api_call(api_func, max_retries=3, initial_delay=2):
    """
    Automatically retry API calls with exponential backoff.
    Only raises exception if all retries fail.
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return api_func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # Don't retry on authentication/quota errors (these won't fix themselves)
            if "authentication" in error_str or "401" in error_str or "403" in error_str or ("invalid" in error_str and "key" in error_str):
                raise Exception("🔐 **API Key Error** - Your API key is invalid or expired. Please check your API key in the code.")
            elif "quota" in error_str or "insufficient" in error_str or "billing" in error_str:
                raise Exception("💳 **Quota Exceeded** - Your OpenAI account has reached its usage limit. Please check your account billing/credits.")
            
            # For rate limits and timeouts, retry with backoff
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff: 2s, 4s, 8s
            else:
                # Last attempt failed, format the error message
                if "rate_limit" in error_str or "429" in error_str or "rate limit" in error_str:
                    raise Exception("⚠️ **Rate Limit Exceeded** - Please wait a moment and try again.")
                elif "timeout" in error_str or "timed out" in error_str:
                    raise Exception("⏱️ **Request Timed Out** - The request took too long. Please try again.")
                else:
                    error_type = type(e).__name__
                    raise Exception(f"❌ **API Error** ({error_type}): {str(e)[:200]}")
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


def initialize_csv_db():
    """Initialize CSV database file with headers if it doesn't exist."""
    if not os.path.exists(CSV_DB_FILE):
        headers = [
            "title", "summary", "format", "genre", "target_audience", 
            "tone", "characters", "themes", "content_hash", "upload_date"
        ]
        with open(CSV_DB_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def save_story_to_csv(filename: str, analysis: dict, content_hash: str):
    """Save story data to CSV database. Data persists permanently on disk."""
    initialize_csv_db()
    
    # Extract data
    story_title = analysis.get("story_title", "").strip()
    # If no title extracted, use filename as fallback
    if not story_title:
        story_title = filename
    summary = analysis.get("brief_summary", "")
    format_type = analysis.get("format_type", "unknown")
    genre = analysis.get("genre", "unknown")
    target_audience = analysis.get("target_audience", "unknown")
    tone = analysis.get("tone", "unknown")
    
    # Format characters as JSON string to preserve descriptions
    characters_list = analysis.get("main_characters", [])
    # Ensure all characters are in dict format with name and description
    formatted_characters = []
    for c in characters_list:
        if isinstance(c, dict):
            formatted_characters.append({
                "name": c.get("name", ""),
                "description": c.get("description", "")
            })
        elif isinstance(c, str):
            formatted_characters.append({
                "name": c,
                "description": ""
            })
    characters_json = json.dumps(formatted_characters)
    
    # Format themes as string (tags)
    themes_list = analysis.get("themes", [])
    themes_str = ", ".join(themes_list)
    
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if story already exists (by content_hash)
    if os.path.exists(CSV_DB_FILE):
        try:
            df = pd.read_csv(CSV_DB_FILE)
            if 'content_hash' in df.columns and content_hash in df['content_hash'].values:
                # Update existing entry
                # Check if new format (with story_title) or old format
                if 'story_title' in df.columns:
                    df.loc[df['content_hash'] == content_hash, [
                        'filename', 'story_title', 'summary', 'format', 'genre', 'target_audience', 
                        'tone', 'characters', 'themes', 'upload_date'
                    ]] = [
                        filename, story_title, summary, format_type, genre, target_audience, 
                        tone, characters_json, themes_str, upload_date
                    ]
                else:
                    # Old format
                    df.loc[df['content_hash'] == content_hash, [
                        'title', 'summary', 'format', 'genre', 'target_audience', 
                        'tone', 'characters', 'themes', 'upload_date'
                    ]] = [
                        story_title, summary, format_type, genre, target_audience, 
                        tone, characters_json, themes_str, upload_date
                    ]
                df.to_csv(CSV_DB_FILE, index=False, encoding='utf-8')
                return
        except Exception as e:
            # If CSV is corrupted, create new one
            pass
    
    # Add new entry
    # Check if CSV has new format (with story_title) or old format
    if os.path.exists(CSV_DB_FILE):
        try:
            df_check = pd.read_csv(CSV_DB_FILE, nrows=0)
            if 'story_title' in df_check.columns:
                # New format
                new_row = {
                    "filename": filename,
                    "story_title": story_title,
                    "summary": summary,
                    "format": format_type,
                    "genre": genre,
                    "target_audience": target_audience,
                    "tone": tone,
                    "characters": characters_json,
                    "themes": themes_str,
                    "content_hash": content_hash,
                    "upload_date": upload_date
                }
            else:
                # Old format - use title field
                new_row = {
                    "title": story_title,
                    "summary": summary,
                    "format": format_type,
                    "genre": genre,
                    "target_audience": target_audience,
                    "tone": tone,
                    "characters": characters_json,
                    "themes": themes_str,
                    "content_hash": content_hash,
                    "upload_date": upload_date
                }
        except Exception:
            # Default to new format if can't read
            new_row = {
                "filename": filename,
                "story_title": story_title,
                "summary": summary,
                "format": format_type,
                "genre": genre,
                "target_audience": target_audience,
                "tone": tone,
                "characters": characters_json,
                "themes": themes_str,
                "content_hash": content_hash,
                "upload_date": upload_date
            }
    else:
        # New file - use new format
        new_row = {
            "filename": filename,
            "story_title": story_title,
            "summary": summary,
            "format": format_type,
            "genre": genre,
            "target_audience": target_audience,
            "tone": tone,
            "characters": characters_json,
            "themes": themes_str,
            "content_hash": content_hash,
            "upload_date": upload_date
        }
    
    df = pd.DataFrame([new_row])
    if os.path.exists(CSV_DB_FILE):
        # Append to existing file
        df.to_csv(CSV_DB_FILE, mode='a', header=False, index=False, encoding='utf-8')
    else:
        # Create new file
        df.to_csv(CSV_DB_FILE, index=False, encoding='utf-8')


def load_stories_from_csv():
    """Load all stories from CSV database. Data persists permanently across app sessions."""
    if not os.path.exists(CSV_DB_FILE):
        return []
    
    try:
        df = pd.read_csv(CSV_DB_FILE)
        # Filter out any empty rows - check for either 'title' (old format) or 'story_title' (new format)
        if 'story_title' in df.columns:
            df = df.dropna(subset=['story_title'])
        elif 'title' in df.columns:
            df = df.dropna(subset=['title'])
        return df.to_dict('records')
    except Exception as e:
        # Silently return empty list if file is corrupted or doesn't exist
        return []


# Initialize CSV database at startup
initialize_csv_db()

client = OpenAI(api_key=API_KEY)

st.set_page_config(page_title="Spotlight Analyzer", page_icon="🎬", layout="wide")
st.title("🎬 Spotlight Story Analyzer & Improvement Tool")
st.markdown("**Upload your story/script files, get comprehensive analysis, improvement feedback, and chat with AI about your stories.**")

# -----------------------------
# Init session state for file contexts & chat
# -----------------------------
if "file_contexts" not in st.session_state:
    st.session_state.file_contexts = {}  # {content_hash: {"filename": ..., "text": ..., "analysis": {...}, "hash": ...}}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # {content_hash: [{"role": "user"/"assistant", "content": "..."}]}

if "file_hash_map" not in st.session_state:
    st.session_state.file_hash_map = {}  # {filename: content_hash} for quick lookup


# -----------------------------
# File Upload (Multiple)
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Script / Story / Concept Files",
    type=["txt", "md", "pdf"] if PDF_SUPPORT else ["txt", "md"],
    accept_multiple_files=True,
)

if not uploaded_files:
    file_types = ".txt, .md" + (", .pdf" if PDF_SUPPORT else "")
    st.info(f"Please upload one or more {file_types} files above.")
    if not PDF_SUPPORT:
        st.warning("⚠️ PDF support not available. Install PyPDF2 or pypdf: `pip install PyPDF2` or `pip install pypdf`")
    st.stop()


# -----------------------------
# Main System Prompt (Classification)
# -----------------------------
CLASSIFICATION_PROMPT = """
You are an expert film development analyst.

Analyze the uploaded file content and classify the following:

1. story_title → Extract the story title if it's explicitly mentioned in the file (look for patterns like "Title:", "Title -", or a title at the beginning). If no explicit title is found, leave this as an empty string.

2. format_type (choose one):
   - feature_film
   - short_film
   - documentary
   - web_series
   - marketing_video
   - reel_or_short

3. genre → e.g. thriller, sci-fi drama, crime, romantic comedy, fantasy adventure

4. target_audience → write specific age & audience segment  
   Example: "Young adults 18-30 urban", "Family audience", "Women 25-40", "Professionals & entrepreneurs"

5. brief_summary → 2-4 sentence summary in simple language

6. main_characters → list of main characters with brief descriptions (as a JSON array of objects with "name" and "description")
   CRITICAL REQUIREMENT: For each character, you MUST provide a meaningful description (2-3 sentences) describing their role, personality traits, relationships, or significance in the story. Base descriptions on actual content from the story text. NEVER use empty strings, "No description", or placeholder text. If you cannot determine details, provide at least a basic description of their role in the story based on what you can infer from the text.

7. themes → list of main themes or motifs in the story (as a JSON array of strings)

8. tone → describe the overall tone/mood (e.g., "dark and suspenseful", "light-hearted and comedic", "dramatic and emotional")

9. reasoning → short explanation on why those categories were chosen

Return output ONLY in JSON format like:

{
  "story_title": "",
  "format_type": "",
  "genre": "",
  "target_audience": "",
  "brief_summary": "",
  "main_characters": [{"name": "", "description": ""}],
  "themes": [""],
  "tone": "",
  "reasoning": ""
}
""".strip()


def analyze_text(text: str) -> dict:
    """
    Send the file text to the LLM to classify format, genre, audience, etc.
    Returns analysis dict or raises exception.
    Automatically retries on rate limits/timeouts.
    """
    if len(text) > 12000:
        text_to_send = text[:12000] + "\n[TRUNCATED FOR ANALYSIS]"
    else:
        text_to_send = text

    user_query = f'Content to analyze:\n""" {text_to_send} """'

    def _make_api_call():
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.3,
            timeout=60,  # 60 second timeout
        )
        return response

    # Automatic retry with exponential backoff
    response = retry_api_call(_make_api_call, max_retries=3, initial_delay=2)

    if not response or not response.choices:
        raise Exception("❌ Empty response from API. Please try again.")

    raw = response.choices[0].message.content
    if not raw:
        raise Exception("❌ No content in API response. Please try again.")

    try:
        result = json.loads(raw)
        # Ensure all new fields have defaults
        if "story_title" not in result:
            result["story_title"] = ""
        if "main_characters" not in result:
            result["main_characters"] = []
        else:
            # Ensure all characters have proper descriptions
            cleaned_characters = []
            for char in result["main_characters"]:
                if isinstance(char, dict):
                    name = char.get("name", "").strip()
                    desc = char.get("description", "").strip()
                    # Only provide default if description is truly missing
                    # Don't overwrite valid descriptions
                    if not desc or desc.strip() == "" or desc.lower() == 'no description':
                        desc = f"Main character named {name if name else 'Unknown'}"
                    if name:  # Only add if name exists
                        cleaned_characters.append({"name": name, "description": desc})
                elif isinstance(char, str) and char.strip():
                    # Convert string to dict format
                    cleaned_characters.append({"name": char.strip(), "description": f"Main character named {char.strip()}"})
            result["main_characters"] = cleaned_characters
        if "themes" not in result:
            result["themes"] = []
        if "tone" not in result:
            result["tone"] = "unknown"
        return result
    except json.JSONDecodeError as e:
        raise Exception(f"❌ Failed to parse API response as JSON: {str(e)[:200]}")


# -----------------------------
# Chat System Prompt (Q&A)
# -----------------------------
CHAT_SYSTEM_PROMPT = """
You are a helpful story assistant.

You will be given:
- The full or partial text of a story/script/brief
- A JSON analysis of that story (format_type, genre, target_audience, summary, reasoning)
- A user question

Your job:
- Answer ONLY using information that can be reasonably inferred from the story text and its analysis.
- Be specific and stay focused on the story content, characters, tone, structure, and themes.
- If the answer is not clear from the story, say you are not sure instead of inventing details.
- Keep answers concise and clear.
""".strip()


def answer_question(file_text: str, analysis: dict, question: str, improvements: dict = None) -> str:
    """
    Use LLM to answer questions about the selected story file.
    Now includes improvements data if available.
    """
    # Truncate story text if very long
    if len(file_text) > 12000:
        story_snippet = file_text[:12000] + "\n[TRUNCATED]"
    else:
        story_snippet = file_text

    analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)
    
    # Include improvements if available
    improvements_section = ""
    if improvements:
        improvements_json = json.dumps(improvements, ensure_ascii=False, indent=2)
        improvements_section = f"""
    
    STORY IMPROVEMENTS & SCORES:
    {improvements_json}
    """

    user_content = textwrap.dedent(f"""
    STORY TEXT:
    \"\"\"
    {story_snippet}
    \"\"\"

    STORY ANALYSIS JSON:
    {analysis_json}{improvements_section}

    USER QUESTION:
    {question}
    """)

    def _make_api_call():
        return client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.4,
            timeout=60,
        )

    # Automatic retry with exponential backoff
    response = retry_api_call(_make_api_call, max_retries=3, initial_delay=2)

    if not response or not response.choices:
        raise Exception("❌ Empty response from API. Please try again.")

    content = response.choices[0].message.content
    if not content:
        raise Exception("❌ No content in API response. Please try again.")

    return content.strip()


# -----------------------------
# Story Improvement & Feedback System
# -----------------------------
IMPROVEMENT_PROMPT = """
You are an expert story consultant and script doctor. Analyze the story and provide constructive feedback.

Provide:
1. strengths → List 3-5 key strengths of the story (as JSON array)
2. weaknesses → List 3-5 areas that need improvement (as JSON array)
3. recommendations → List 5-7 specific, actionable recommendations to improve the story (as JSON array of objects with "area" and "suggestion")
4. story_structure_score → Score out of 10 for narrative structure
5. story_structure_explanation → Brief one-line explanation of why this score was given for this specific story
6. character_development_score → Score out of 10 for character depth and development
7. character_development_explanation → Brief one-line explanation of why this score was given for this specific story
8. dialogue_quality_score → Score out of 10 for dialogue quality
9. dialogue_quality_explanation → Brief one-line explanation of why this score was given for this specific story
10. pacing_score → Score out of 10 for story pacing
11. pacing_explanation → Brief one-line explanation of why this score was given for this specific story
12. originality_score → Score out of 10 for originality and uniqueness
13. originality_explanation → Brief one-line explanation of why this score was given for this specific story
14. marketability_score → Score out of 10 for commercial viability
15. marketability_explanation → Brief one-line explanation of why this score was given for this specific story
16. overall_score → Overall quality score out of 10
17. improvement_priority → List top 3 priority areas to focus on (as JSON array)

Return output ONLY in JSON format.
""".strip()


def get_story_improvements(text: str, analysis: dict) -> dict:
    """
    Get detailed feedback and improvement suggestions for the story.
    Returns improvements dict or raises exception.
    """
    if len(text) > 12000:
        text_to_send = text[:12000] + "\n[TRUNCATED FOR ANALYSIS]"
    else:
        text_to_send = text

    analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)

    user_content = textwrap.dedent(f"""
    STORY TEXT:
    \"\"\"
    {text_to_send}
    \"\"\"

    INITIAL ANALYSIS:
    {analysis_json}

    Provide detailed feedback and improvement suggestions.
    """)

    def _make_api_call():
        return client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": IMPROVEMENT_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.5,
            timeout=60,
        )

    # Automatic retry with exponential backoff
    response = retry_api_call(_make_api_call, max_retries=3, initial_delay=2)

    if not response or not response.choices:
        raise Exception("❌ Empty response from API. Please try again.")

    raw = response.choices[0].message.content
    if not raw:
        raise Exception("❌ No content in API response. Please try again.")

    try:
        result = json.loads(raw)
        # Ensure all fields have defaults
        defaults = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "story_structure_score": 5,
            "character_development_score": 5,
            "dialogue_quality_score": 5,
            "pacing_score": 5,
            "originality_score": 5,
            "marketability_score": 5,
            "overall_score": 5,
            "improvement_priority": []
        }
        for key, default_value in defaults.items():
            if key not in result:
                result[key] = default_value
        return result
    except json.JSONDecodeError as e:
        raise Exception(f"❌ Failed to parse API response as JSON: {str(e)[:200]}")


# -----------------------------
# Story Similarity Search Function
# -----------------------------
SIMILARITY_PROMPT = """
You are an expert story analyst. Compare a user's story query with a database story and determine similarity.

Given:
1. User's story query/idea/description
2. A story from the database (with its analysis)

Provide:
1. similarity_score → A percentage (0-100) indicating how similar the stories are
2. reasoning → Brief explanation of why they are similar or different
3. key_similarities → List of main similarities (as JSON array)
4. key_differences → List of main differences (as JSON array)

Consider: genre, themes, plot elements, character types, tone, setting, target audience, format type.

Return output ONLY in JSON format:
{
  "similarity_score": 0-100,
  "reasoning": "",
  "key_similarities": [""],
  "key_differences": [""]
}
""".strip()


TITLE_GENERATION_PROMPT = """
You are a creative story title generator. Based on a story idea or description, generate compelling title suggestions.

Given a user's story idea/description, create:
1. titles → List of 5-7 creative, engaging title suggestions (as JSON array)
2. reasoning → Brief explanation of why these titles fit the story idea

Consider:
- Genre and tone
- Key themes and elements
- Target audience appeal
- Marketability and memorability
- Different title styles (descriptive, metaphorical, character-based, etc.)

Return output ONLY in JSON format:
{
  "titles": ["Title 1", "Title 2", ...],
  "reasoning": "Explanation of title choices"
}
""".strip()


def generate_story_titles(story_idea: str, existing_titles_context: str = "") -> dict:
    """
    Generate title suggestions for a story idea.
    Can optionally include context about existing titles from database.
    Returns dict with titles list and reasoning.
    Automatically retries on rate limits/timeouts.
    """
    user_content = f"""
                STORY IDEA:
                {story_idea}
                
                Generate creative title suggestions for this story.
                """
    
    if existing_titles_context:
        user_content += f"""
                
                NOTE: The following titles already exist in the database (use as inspiration, but create NEW unique titles):
                {existing_titles_context}
                """
    
    def _make_api_call():
        return client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": TITLE_GENERATION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,  # More creative for title generation
            timeout=30,
        )
    
    try:
        # Automatic retry with exponential backoff
        response = retry_api_call(_make_api_call, max_retries=3, initial_delay=2)
        
        if not response or not response.choices:
            return {"titles": [], "reasoning": "Could not generate titles."}
        
        raw = response.choices[0].message.content
        if not raw:
            return {"titles": [], "reasoning": "No content in response."}
        
        result = json.loads(raw)
        return {
            "titles": result.get("titles", []),
            "reasoning": result.get("reasoning", "Title suggestions generated.")
        }
    except Exception as e:
        return {
            "titles": [],
            "reasoning": f"Error generating titles: {str(e)[:100]}"
        }


def extract_story_title_from_summary(summary: str, filename: str) -> str:
    """
    Extract or generate a meaningful story title from the summary.
    Falls back to filename if summary is empty.
    """
    if not summary or not summary.strip():
        return filename
    
    # Try to extract a title from the first sentence of the summary
    # Look for patterns like "Title: ..." or use first few words
    summary_clean = summary.strip()
    
    # If summary starts with a character name followed by comma, use that as title
    first_sentence = summary_clean.split('.')[0] if '.' in summary_clean else summary_clean
    first_sentence = first_sentence.strip()
    
    # If first sentence is too long, take first few words
    words = first_sentence.split()
    if len(words) > 8:
        # Take first 6-8 words
        title = ' '.join(words[:8])
        if len(title) > 60:
            title = title[:57] + "..."
        return title
    elif len(first_sentence) > 60:
        return first_sentence[:57] + "..."
    else:
        return first_sentence


def suggest_existing_titles_from_db(story_idea: str, top_n: int = 5) -> list:
    """
    Suggest existing story titles from the database that match the story idea.
    Uses LLM to calculate relevance for better matching.
    Returns list of titles with similarity scores.
    """
    stories = load_stories_from_csv()
    
    if not stories:
        return []
    
    results = []
    
    for story in stories:
        # Check for new format (with story_title and filename) or old format
        if "story_title" in story and "filename" in story:
            # New format
            story_title = story.get("story_title", "").strip()
            filename = story.get("filename", "Unknown")
            if not story_title:
                story_title = filename
        else:
            # Old format - the "title" field contains the actual story title
            story_title = story.get("title", "").strip()
            filename = story_title if story_title else "Unknown"
            # Only extract from summary if title looks like a filename (ends with extension)
            # Otherwise, use the title directly as it's already the story title
            if story_title and (story_title.endswith('.txt') or story_title.endswith('.md') or story_title.endswith('.pdf')):
                summary_temp = story.get("summary", "")
                story_title = extract_story_title_from_summary(summary_temp, story_title)
        
        summary = story.get("summary", "")
        genre = story.get("genre", "unknown")
        themes = story.get("themes", "")
        format_type = story.get("format", "unknown")
        
        # Prepare story context for LLM comparison
        story_context = f"""
        Title: {story_title}
        Summary: {summary}
        Genre: {genre}
        Format: {format_type}
        Themes: {themes}
        """
        
        if len(story_context) > 1000:
            story_context = story_context[:1000] + "..."
        
        try:
            # Use LLM to calculate relevance for title matching
            def _make_api_call():
                return client.chat.completions.create(
                    model=MODEL_NAME,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are a story title matching expert. Compare a user's story idea with an existing story title and determine how relevant the title is for the user's idea. Return a relevance score (0-100) and brief reasoning."},
                        {"role": "user", "content": f"""
                        USER'S STORY IDEA:
                        {story_idea}
                        
                        EXISTING STORY IN DATABASE:
                        {story_context}
                        
                        How relevant is the title "{story_title}" for the user's story idea? 
                        Consider: genre match, theme similarity, plot elements, target audience.
                        
                        Return JSON with:
                        {{
                          "relevance_score": 0-100,
                          "reasoning": "Brief explanation"
                        }}
                        """},
                    ],
                    temperature=0.3,
                    timeout=20,
                )
            
            response = retry_api_call(_make_api_call, max_retries=2, initial_delay=1)
            
            if response and response.choices:
                match_data = json.loads(response.choices[0].message.content)
                relevance = match_data.get("relevance_score", 0)
                reasoning = match_data.get("reasoning", "")
                
                # Only include titles with meaningful relevance (>= 15%)
                if relevance >= 15:
                    results.append({
                        "title": story_title,
                        "filename": filename,
                        "relevance": relevance,
                        "genre": genre,
                        "format": format_type,
                        "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                        "themes": themes,
                        "reasoning": reasoning
                    })
        except Exception:
            # Fallback to simple keyword matching if LLM fails
            story_context_lower = f"{story_title} {summary} {genre} {themes}".lower()
            idea_lower = story_idea.lower()
            
            idea_words = set(idea_lower.split())
            context_words = set(story_context_lower.split())
            common_words = idea_words.intersection(context_words)
            
            if common_words:
                relevance = len(common_words) / max(len(idea_words), 1) * 100
                if relevance >= 15:
                    results.append({
                        "title": story_title,
                        "filename": filename,  # Keep filename for reference
                        "relevance": min(relevance, 100),
                        "genre": genre,
                        "format": format_type,
                        "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                        "themes": themes,
                        "reasoning": "Matched based on keywords"
                    })
    
    # Sort by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:top_n]


def find_similar_stories(query: str, top_n: int = None) -> list:
    """
    Find similar stories from CSV database based on natural language query.
    Returns ranked list of stories sorted by similarity score and tags.
    """
    # Load stories from CSV database
    stories = load_stories_from_csv()
    
    if not stories:
        return []
    
    results = []
    
    for story in stories:
        title = story.get("title", "Unknown")
        summary = story.get("summary", "")
        format_type = story.get("format", "unknown")
        genre = story.get("genre", "unknown")
        target_audience = story.get("target_audience", "unknown")
        tone = story.get("tone", "unknown")
        themes = story.get("themes", "")  # Tags
        characters = story.get("characters", "")
        
        # Prepare story summary for comparison
        story_summary = f"""
        Title: {title}
        Format: {format_type}
        Genre: {genre}
        Target Audience: {target_audience}
        Tone: {tone}
        Summary: {summary}
        Tags/Themes: {themes}
        Characters: {characters}
        """
        
        # Truncate if needed
        if len(story_summary) > 2000:
            story_summary = story_summary[:2000] + "..."
        
        try:
            # Get similarity score from LLM (focusing on tags/themes for ranking)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SIMILARITY_PROMPT},
                    {"role": "user", "content": f"""
                    USER QUERY:
                    {query}
                    
                    DATABASE STORY:
                    {story_summary}
                    
                    Calculate similarity score (0-100) based on:
                    1. Tags/Themes matching
                    2. Genre compatibility
                    3. Plot elements similarity
                    4. Character types
                    5. Overall story concept
                    
                    Provide similarity score and reasoning.
                    """},
                ],
                temperature=0.3,
                timeout=30,
            )
            
            similarity_data = json.loads(response.choices[0].message.content)
            similarity_score = similarity_data.get("similarity_score", 0)
            
            # Only include stories with meaningful similarity (>= 20%)
            if similarity_score >= 20:
                results.append({
                    "title": title,
                    "filename": title,
                    "similarity": similarity_score,
                    "format": format_type.replace("_", " ").title(),
                    "genre": genre,
                    "summary": summary,
                    "characters": characters.split(", ") if characters else [],
                    "tags": themes.split(", ") if themes else [],
                    "reasoning": similarity_data.get("reasoning", "Similar story found"),
                    "similarities": similarity_data.get("key_similarities", []),
                    "differences": similarity_data.get("key_differences", []),
                })
        except Exception:
            # Skip stories that fail similarity check
            continue
    
    # Sort by similarity score (highest first) - Ranking Pipeline
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top N results if specified, otherwise return top matches
    if top_n:
        return results[:top_n]
    
    # Return all results above threshold (up to reasonable limit)
    return results[:20]  # Max 20 results


# -----------------------------
# MAIN TABS: Analysis, Chat, Story Search
# -----------------------------
main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Analysis Results", "💬 Chat & Questions", "🔍 Story Search & Filter"])

with main_tab1:
    st.header("🔍 Analysis Results")

# Display stats if files exist
if st.session_state.file_contexts:
    total_files = len(st.session_state.file_contexts)
    files_with_improvements = sum(
        1 for f in st.session_state.file_contexts.values() 
        if f.get("improvements")
    )
    total_chat_messages = sum(len(chats) for chats in st.session_state.chat_history.values())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Files Analyzed", total_files)
    with col2:
        st.metric("💡 With Feedback", files_with_improvements)
    with col3:
        st.metric("💬 Chat Messages", total_chat_messages)
    with col4:
        avg_score = "N/A"
        scores = []
        for f in st.session_state.file_contexts.values():
            improvements = f.get("improvements")
            if improvements is not None and isinstance(improvements, dict):
                score = improvements.get("overall_score", 0)
                if score > 0:
                    scores.append(score)
        if scores:
            avg_score = f"{sum(scores) / len(scores):.1f}/10"
        st.metric("⭐ Avg Score", avg_score)
    
    st.markdown("---")

for file in uploaded_files:
    filename = file.name

    # Read file text (handle PDF, TXT, MD)
    try:
        if filename.lower().endswith('.pdf'):
            if not PDF_SUPPORT:
                st.error(f"❌ PDF support not available. Please install PyPDF2: `pip install PyPDF2`")
                continue
            
            # Extract text from PDF
            pdf_file = io.BytesIO(file.read())
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                if not text.strip():
                    st.warning(f"⚠️ PDF `{filename}` appears to have no extractable text. It may be image-based or encrypted.")
                    continue
            except Exception as pdf_error:
                st.error(f"❌ Cannot read PDF `{filename}`. Error: {str(pdf_error)[:100]}")
                st.info("💡 The PDF may be encrypted, corrupted, or image-based (OCR required).")
                continue
        else:
            # Read text/markdown files
            text = file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"❌ Cannot read file `{filename}`. Error: {str(e)[:100]}")
        continue

    if not text.strip():
        st.warning(f"⚠️ File `{filename}` appears to be empty or unreadable. Skipping.")
        continue

    # Generate content hash for caching
    content_hash = get_content_hash(text)
    
    # Check if this exact content has been analyzed before (by hash)
    if content_hash in st.session_state.file_contexts:
        # Content already analyzed, use cached data
        cached_data = st.session_state.file_contexts[content_hash]
        
        # Check if previous attempt failed
        if cached_data.get("analysis_error"):
            # Previous attempt failed - skip to avoid repeated errors
            st.warning(f"⚠️ `{filename}` previously failed analysis. Skipping to avoid repeated errors.")
            st.info("💡 Please wait a moment and refresh the page, or check your API key/connection.")
            continue
        
        analysis = cached_data["analysis"]
        improvements = cached_data.get("improvements")
        # Don't show info message if already displayed
    else:
        # Check CSV database first to see if story was already analyzed
        csv_stories = load_stories_from_csv()
        csv_match = None
        for story in csv_stories:
            if story.get("content_hash") == content_hash:
                csv_match = story
                break
        
        if csv_match:
            # Story exists in CSV - reconstruct analysis from CSV data
            # Parse characters - try JSON first, fallback to comma-separated names
            characters_data = csv_match.get("characters", "")
            try:
                # Try to parse as JSON (new format with descriptions)
                main_characters = json.loads(characters_data) if characters_data else []
            except (json.JSONDecodeError, TypeError):
                # Fallback to old format (comma-separated names)
                main_characters = [{"name": c.strip(), "description": ""} for c in characters_data.split(",") if c.strip()]
            
            analysis = {
                "format_type": csv_match.get("format", "unknown"),
                "genre": csv_match.get("genre", "unknown"),
                "target_audience": csv_match.get("target_audience", "unknown"),
                "brief_summary": csv_match.get("summary", ""),
                "main_characters": main_characters,
                "themes": [t.strip() for t in csv_match.get("themes", "").split(",") if t.strip()],
                "tone": csv_match.get("tone", "unknown"),
                "reasoning": f"Loaded from database - {csv_match.get('upload_date', '')}"
            }
            improvements = None
            
            # Store in session state
            st.session_state.file_contexts[content_hash] = {
                "filename": filename,
                "text": text,
                "analysis": analysis,
                "improvements": improvements,
                "hash": content_hash,
            }
            st.session_state.file_hash_map[filename] = content_hash
            st.info(f"ℹ️ Loaded `{filename}` from database (previously analyzed)")
        else:
            # New content, need to analyze
            with st.spinner(f"🔍 Analyzing `{filename}`..."):
                try:
                    analysis = analyze_text(text)
                    improvements = None  # Will be generated on demand
                    
                    # Store with content hash as key
                    st.session_state.file_contexts[content_hash] = {
                        "filename": filename,
            "text": text,
            "analysis": analysis,
                        "improvements": improvements,
                        "hash": content_hash,
                        "analysis_error": False,  # Mark as successfully analyzed
                    }
                    
                    # Also maintain filename -> hash mapping for quick lookup
                    st.session_state.file_hash_map[filename] = content_hash
                    
                    # Save to CSV database
                    try:
                        save_story_to_csv(filename, analysis, content_hash)
                    except Exception as e:
                        st.warning(f"⚠️ Could not save to database: {str(e)[:100]}")
                    
                except Exception as e:
                    # Cache the failed attempt to prevent immediate retry
                    error_msg = str(e)
                    error_lower = error_msg.lower()
                    
                    st.session_state.file_contexts[content_hash] = {
                        "filename": filename,
                        "text": text,
                        "analysis": {},  # Empty analysis for failed attempts
                        "improvements": None,
                        "hash": content_hash,
                        "analysis_error": True,  # Mark as failed
                        "error_message": error_msg,
                    }
                    st.session_state.file_hash_map[filename] = content_hash
                    
                    # Show appropriate message based on error type
                    if "quota" in error_lower or "billing" in error_lower:
                        st.error(f"❌ **Quota Exceeded** - Your OpenAI account has reached its usage limit.")
                        st.info("💡 **To fix this:** Check your OpenAI account billing/credits at https://platform.openai.com/usage. The file will be shown below, but analysis requires an active API quota.")
                    elif "authentication" in error_lower or "api key" in error_lower:
                        st.error(f"❌ **API Key Error** - Your API key is invalid or expired.")
                        st.info("💡 **To fix this:** Update your API key in the code. The file will be shown below.")
                    else:
                        st.warning(f"⚠️ Analysis failed for `{filename}`: {error_msg}")
                        st.info("💡 The file will be shown below. Please try again later.")
    
    # Get the stored data for this hash (get fresh reference)
    file_data = st.session_state.file_contexts.get(content_hash)
    if not file_data:
        # File not in session state - skip (shouldn't happen, but safety check)
        continue
    
    analysis = file_data.get("analysis", {})
    improvements = file_data.get("improvements")
    analysis_error = file_data.get("analysis_error", False)
    error_message = file_data.get("error_message", "")

    st.subheader(f"📄 {filename}")
    
    # Initialize active tab for this file
    tab_key = f"active_tab_{content_hash}"
    if tab_key not in st.session_state:
        st.session_state[tab_key] = "📊 Overview"
    
    # Use radio buttons for tab selection (allows programmatic control)
    selected_tab = st.radio(
        "Select view:",
        ["📊 Overview", "💡 Improvement Feedback", "📈 Story Scores", "💾 Export"],
        horizontal=True,
        key=f"tab_selector_{content_hash}",
        index=["📊 Overview", "💡 Improvement Feedback", "📈 Story Scores", "💾 Export"].index(st.session_state[tab_key]) if st.session_state[tab_key] in ["📊 Overview", "💡 Improvement Feedback", "📈 Story Scores", "💾 Export"] else 0
    )
    st.session_state[tab_key] = selected_tab
    
    st.markdown("---")

    if selected_tab == "📊 Overview":
        # Check if analysis failed (only show if all retries exhausted)
        if analysis_error or not analysis:
            error_lower = error_message.lower() if error_message else ""
            
            if "quota" in error_lower or "billing" in error_lower:
                st.error(f"❌ **Quota Exceeded** - Your OpenAI account has reached its usage limit.")
                st.info("💡 **To fix:** Check your OpenAI account billing/credits at https://platform.openai.com/usage")
            elif "authentication" in error_lower or "api key" in error_lower:
                st.error(f"❌ **API Key Error** - Your API key is invalid or expired.")
                st.info("💡 **To fix:** Update your API key in the code.")
            else:
                st.error(f"❌ **Analysis Failed:** {error_message if error_message else 'Could not analyze this file'}")
                st.info("💡 Please check your API key, internet connection, and try uploading the file again.")
            
            st.markdown("---")
            # Show file preview even if analysis failed
            st.markdown("**📄 File Preview (first 2000 characters):**")
            st.text(text[:2000] + "..." if len(text) > 2000 else text)
            st.markdown("---")
        else:
            # Normal analysis display
            # Format type badge
            format_type = analysis.get("format_type", "unknown")
            format_colors = {
                "feature_film": "🔴",
                "short_film": "🟢",
                "documentary": "🔵",
                "web_series": "🟡",
                "marketing_video": "🟣",
                "reel_or_short": "🟠"
            }
            format_icon = format_colors.get(format_type, "⚪")
            st.markdown(f"{format_icon} **Format:** {format_type.replace('_', ' ').title()}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Genre**")
                st.write(analysis.get("genre", "unknown"))
            with col2:
                st.markdown("**Target Audience**")
                st.write(analysis.get("target_audience", "unknown"))
            with col3:
                st.markdown("**Tone**")
                st.write(analysis.get("tone", "unknown"))

            st.markdown("**📘 Brief Summary**")
            st.write(analysis.get("brief_summary", "-"))

            # Main Characters
            characters = analysis.get("main_characters", [])
            if characters:
                st.markdown("**👥 Main Characters**")
                for char in characters:
                    if isinstance(char, dict):
                        char_name = char.get('name', 'Unknown')
                        char_desc = char.get('description', '').strip()
                        
                        # If description is missing or empty, show a message
                        if not char_desc or char_desc.lower() in ['no description', '']:
                            st.markdown(f"- **{char_name}**: *Description not available*")
                            st.caption("💡 Character description was not provided in the analysis.")
                        else:
                            st.markdown(f"- **{char_name}**: {char_desc}")
                    else:
                        st.markdown(f"- {char}")

            # Themes
            themes = analysis.get("themes", [])
            if themes:
                st.markdown("**🎭 Themes**")
                theme_badges = " ".join([f"`{theme}`" for theme in themes if theme])
                st.markdown(theme_badges)

            st.markdown("**🧠 Reasoning**")
            st.write(analysis.get("reasoning", "-"))

            with st.expander("🔍 View Text Preview (first 1500 characters)"):
                st.code(text[:1500])

    elif selected_tab == "💡 Improvement Feedback":
        st.markdown("### 💡 Story Improvement & Feedback")
        
        # Get fresh data reference
        current_file_data = st.session_state.file_contexts[content_hash]
        improvements = current_file_data.get("improvements")
        improvement_error = current_file_data.get("improvement_error", False)
        improvement_error_msg = current_file_data.get("improvement_error_msg", "")
        
        # Check if improvements already exist
        if improvements:
            # Improvements already generated - show them
            pass
        elif improvement_error:
            # Previous attempt failed - show retry option (don't show error again)
            st.warning(f"⚠️ Previous attempt failed: {improvement_error_msg}")
            st.info("💡 Please wait a moment before retrying, or check your API key and connection.")
            if st.button("🔄 Retry Improvement Analysis", key=f"retry_improve_{filename}", type="primary"):
                # Clear error flag before retry
                st.session_state.file_contexts[content_hash]["improvement_error"] = False
                st.session_state.file_contexts[content_hash]["improvement_error_msg"] = ""
                # Preserve the active tab
                st.session_state[tab_key] = "💡 Improvement Feedback"
                st.rerun()
        else:
            # No improvements yet, show generate button
            if st.button("🔄 Generate Improvement Analysis", key=f"improve_{filename}", type="primary"):
                with st.spinner("🔍 Analyzing story for improvements..."):
                    try:
                        improvements = get_story_improvements(text, analysis)
                        # Success - save improvements and clear any error flags
                        st.session_state.file_contexts[content_hash]["improvements"] = improvements
                        st.session_state.file_contexts[content_hash]["improvement_error"] = False
                        st.session_state.file_contexts[content_hash]["improvement_error_msg"] = ""
                        # Preserve the active tab (stay on Improvement Feedback)
                        st.session_state[tab_key] = "💡 Improvement Feedback"
                        st.success("✅ Improvement analysis generated successfully!")
                        st.rerun()
                    except Exception as e:
                        # Cache the error to prevent immediate retry
                        error_msg = str(e)
                        st.session_state.file_contexts[content_hash]["improvement_error"] = True
                        st.session_state.file_contexts[content_hash]["improvement_error_msg"] = error_msg
                        # Preserve the active tab
                        st.session_state[tab_key] = "💡 Improvement Feedback"
                        st.error(f"❌ Failed to generate improvements: {error_msg}")
                        st.info("💡 Please check your API key, internet connection, and try again. Wait a moment before retrying.")
                        st.rerun()  # Rerun to show retry button instead of generate button
        
        # Get fresh improvements after potential update
        improvements = st.session_state.file_contexts[content_hash].get("improvements")
        
        if improvements:
            # Strengths
            strengths = improvements.get("strengths", [])
            if strengths:
                st.markdown("**✅ Key Strengths**")
                for strength in strengths:
                    st.success(f"✓ {strength}")
            
            # Weaknesses
            weaknesses = improvements.get("weaknesses", [])
            if weaknesses:
                st.markdown("**⚠️ Areas for Improvement**")
                for weakness in weaknesses:
                    st.warning(f"⚠ {weakness}")
            
            # Recommendations
            recommendations = improvements.get("recommendations", [])
            if recommendations:
                st.markdown("**💡 Actionable Recommendations**")
                for idx, rec in enumerate(recommendations, 1):
                    if isinstance(rec, dict):
                        area = rec.get("area", "General")
                        suggestion = rec.get("suggestion", rec.get("recommendation", ""))
                        st.markdown(f"**{idx}. {area}**")
                        st.info(f"💡 {suggestion}")
                    else:
                        st.info(f"💡 {rec}")
            
            # Priority areas
            priorities = improvements.get("improvement_priority", [])
            if priorities:
                st.markdown("**🎯 Top Priority Areas**")
                for idx, priority in enumerate(priorities, 1):
                    st.markdown(f"{idx}. **{priority}**")
        else:
            st.info("👆 Click the button above to generate detailed improvement feedback for your story.")

    elif selected_tab == "📈 Story Scores":
        st.markdown("### 📈 Story Quality Scores")
        
        improvements = file_data.get("improvements")
        
        if improvements:
            # Overall score with explanation
            overall = improvements.get("overall_score", 0)
            
            # Overall score explanation
            if overall >= 8:
                overall_explanation = "**Excellent** - Your story is well-crafted with strong elements across all areas. It's ready for professional review or production consideration."
            elif overall >= 6:
                overall_explanation = "**Good** - Your story has solid foundations with some strong points. With targeted improvements, it can reach excellent quality."
            elif overall >= 4:
                overall_explanation = "**Fair** - Your story shows promise but needs significant work in several areas. Focus on the priority improvement areas."
            else:
                overall_explanation = "**Needs Work** - Your story requires substantial revision. Review the detailed feedback and focus on fundamental storytelling elements."
            
            st.markdown(f"### Overall Score: {overall}/10")
            st.info(f"💡 **What this means:** {overall_explanation}")
            st.markdown("---")

            # Score bars with summaries
            scores = {
                "Story Structure": {
                    "score": improvements.get("story_structure_score", 0),
                    "summary": "Measures plot organization, narrative arc, and scene progression"
                },
                "Character Development": {
                    "score": improvements.get("character_development_score", 0),
                    "summary": "Evaluates character depth, growth, motivations, and relatability"
                },
                "Dialogue Quality": {
                    "score": improvements.get("dialogue_quality_score", 0),
                    "summary": "Assesses how natural, engaging, and purposeful your dialogue is"
                },
                "Pacing": {
                    "score": improvements.get("pacing_score", 0),
                    "summary": "Measures the rhythm, flow, and balance of tension and release"
                },
                "Originality": {
                    "score": improvements.get("originality_score", 0),
                    "summary": "Evaluates uniqueness and fresh perspective in concept and execution"
                },
                "Marketability": {
                    "score": improvements.get("marketability_score", 0),
                    "summary": "Commercial viability and audience appeal potential"
                },
            }
            
            for metric, data in scores.items():
                score = data["score"]
                summary = data["summary"]
                
                # Get story-specific explanation from improvements
                # Map metric names to explanation field names
                explanation_map = {
                    "Story Structure": "story_structure_explanation",
                    "Character Development": "character_development_explanation",
                    "Dialogue Quality": "dialogue_quality_explanation",
                    "Pacing": "pacing_explanation",
                    "Originality": "originality_explanation",
                    "Marketability": "marketability_explanation"
                }
                explanation_key = explanation_map.get(metric, "")
                story_explanation = improvements.get(explanation_key, "").strip() if explanation_key else ""
                
                # Score interpretation
                if score >= 8:
                    interpretation = "🟢 Excellent"
                elif score >= 6:
                    interpretation = "🟡 Good"
                elif score >= 4:
                    interpretation = "🟠 Fair"
                else:
                    interpretation = "🔴 Needs Improvement"
                
                st.markdown(f"**{metric}** - {interpretation}")
                st.progress(score / 10)
                st.caption(f"{score}/10")
                st.write(f"📝 **Description:** {summary}")
                # Show story-specific explanation if available
                if story_explanation:
                    st.write(f"💡 **Rating Explanation:** {story_explanation}")
            
            # Visual comparison
            st.markdown("**📊 Score Breakdown**")
            score_data = [data["score"] for data in scores.values()]
            score_labels = list(scores.keys())
            
            # Create a simple bar chart using columns
            max_score = max(score_data) if score_data else 10
            for label, data in scores.items():
                score = data["score"]
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.write(f"{label}:")
                with col2:
                    bar = "█" * int((score / 10) * 20)
                    st.write(f"{bar} {score}/10")
            else:
                st.info("👆 Generate improvement analysis first to see story scores.")
                st.markdown("**💡 Tip:** Click on the **💡 Improvement Feedback** view above and click 'Generate Improvement Analysis' to get detailed scores and feedback.")

    elif selected_tab == "💾 Export":
        st.markdown("### 💾 Export Analysis")
        
        # Export JSON
        export_data = {
            "filename": filename,
            "analysis_date": datetime.now().isoformat(),
            "analysis": analysis,
            "improvements": file_data.get("improvements"),
            "text_length": len(text),
            "content_hash": content_hash,
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download Analysis as JSON",
            data=json_str,
            file_name=f"{filename}_analysis_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            key=f"export_json_{filename}"
        )
        
        # Export as formatted text
        text_export = f"""
STORY ANALYSIS REPORT
{'=' * 50}
Filename: {filename}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

BASIC INFORMATION
-----------------
Format: {analysis.get('format_type', 'unknown')}
Genre: {analysis.get('genre', 'unknown')}
Target Audience: {analysis.get('target_audience', 'unknown')}
Tone: {analysis.get('tone', 'unknown')}

SUMMARY
-------
{analysis.get('brief_summary', '-')}

CHARACTERS
----------
"""
        for char in analysis.get('main_characters', []):
            if isinstance(char, dict):
                text_export += f"- {char.get('name', 'Unknown')}: {char.get('description', '')}\n"
            else:
                text_export += f"- {char}\n"
        
        text_export += f"\nTHEMES\n------\n"
        for theme in analysis.get('themes', []):
            text_export += f"- {theme}\n"
        
        improvements = file_data.get("improvements")
        if improvements:
            text_export += f"\n\nSTORY SCORES\n------------\n"
            text_export += f"Overall Score: {improvements.get('overall_score', 0)}/10\n"
            text_export += f"Story Structure: {improvements.get('story_structure_score', 0)}/10\n"
            text_export += f"Character Development: {improvements.get('character_development_score', 0)}/10\n"
            text_export += f"Dialogue Quality: {improvements.get('dialogue_quality_score', 0)}/10\n"
            text_export += f"Pacing: {improvements.get('pacing_score', 0)}/10\n"
            text_export += f"Originality: {improvements.get('originality_score', 0)}/10\n"
            text_export += f"Marketability: {improvements.get('marketability_score', 0)}/10\n"
            
            text_export += f"\n\nIMPROVEMENT FEEDBACK\n--------------------\n"
            text_export += f"\nStrengths:\n"
            for strength in improvements.get('strengths', []):
                text_export += f"- {strength}\n"
            text_export += f"\nAreas for Improvement:\n"
            for weakness in improvements.get('weaknesses', []):
                text_export += f"- {weakness}\n"
            text_export += f"\nRecommendations:\n"
            for rec in improvements.get('recommendations', []):
                if isinstance(rec, dict):
                    text_export += f"- {rec.get('area', 'General')}: {rec.get('suggestion', '')}\n"
                else:
                    text_export += f"- {rec}\n"
        
        st.download_button(
            label="📄 Download Analysis as Text",
            data=text_export,
            file_name=f"{filename}_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            key=f"export_txt_{filename}"
        )
        
        st.markdown("---")
        st.markdown("**Preview:**")
        with st.expander("View Export Preview"):
            st.text(text_export[:2000] + "..." if len(text_export) > 2000 else text_export)

    st.markdown("---")

with main_tab2:
    # -----------------------------
    # CHAT SECTION
    # -----------------------------
    st.header("💬 Chat & Questions")
    
    if st.session_state.file_contexts:
        # Create file options from hash map (show filenames)
        file_options = []
        file_hash_map = {}
        for content_hash, file_data in st.session_state.file_contexts.items():
            fn = file_data.get("filename", f"File_{content_hash[:8]}")
            file_options.append(fn)
            file_hash_map[fn] = content_hash
        
        if not file_options:
            st.info("Upload at least one file to enable chat.")
            st.stop()
        
        selected_file = st.selectbox(
            "Choose a file to chat about:",
            file_options,
            key="chat_file_selectbox"
        )
        
        # Get content hash for selected file
        selected_hash = file_hash_map[selected_file]
        selected_file_data = st.session_state.file_contexts[selected_hash]

        # Initialize chat history for this file (using hash)
        if selected_hash not in st.session_state.chat_history:
            st.session_state.chat_history[selected_hash] = []

        # Clear chat button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history[selected_hash] = []
                st.rerun()

        # Show past chat messages for this file with better UI
        chat_history = st.session_state.chat_history[selected_hash]
        if chat_history:
            for msg in chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        else:
            st.info("💡 Start a conversation by asking a question below or using the suggested questions.")

        # Suggested questions - enhanced with improvement-focused questions
        suggested_questions = [
            "How many characters are there in the story?",
            "What is the main conflict?",
            "What are the key themes?",
            "Who are the main characters?",
            "What is the story's tone?",
            "What is the story structure?",
            "How can I improve the character development?",
            "What are the story's weaknesses?",
            "How can I make the dialogue better?",
            "What should I focus on to improve pacing?",
        ]
        
        st.markdown("**💡 Suggested Questions:**")
        cols = st.columns(2)
        for idx, question in enumerate(suggested_questions):
            with cols[idx % 2]:
                if st.button(question, key=f"suggest_{idx}", use_container_width=True):
                    # Add user message
                    st.session_state.chat_history[selected_hash].append(
                        {"role": "user", "content": question}
                    )

                    file_text = selected_file_data["text"]
                    analysis = selected_file_data["analysis"]
                    improvements = selected_file_data.get("improvements")

                    with st.spinner("💭 Thinking..."):
                        try:
                            answer = answer_question(file_text, analysis, question, improvements)
                        except Exception as e:
                            answer = f"❌ Error: {str(e)}\n\n💡 Please check your API key, internet connection, and try again."
                            st.error(f"Failed to get answer: {str(e)}")

                    # Add assistant message
                    st.session_state.chat_history[selected_hash].append(
                        {"role": "assistant", "content": answer}
                    )

                    # Rerun to refresh UI with new messages
                    st.rerun()

        # Chat input with form to support Enter key submission
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question about this story:", key="chat_input")
            submitted = st.form_submit_button("Ask", type="primary")

            if submitted and user_question.strip():
                # Add user message
                st.session_state.chat_history[selected_hash].append(
                    {"role": "user", "content": user_question}
                )

                file_text = selected_file_data["text"]
                analysis = selected_file_data["analysis"]
                improvements = selected_file_data.get("improvements")

                with st.spinner("💭 Thinking..."):
                    try:
                        answer = answer_question(file_text, analysis, user_question, improvements)
                    except Exception as e:
                        answer = f"❌ Error: {str(e)}\n\n💡 Please check your API key, internet connection, and try again."
                        st.error(f"Failed to get answer: {str(e)}")

                # Add assistant message
                st.session_state.chat_history[selected_hash].append(
                    {"role": "assistant", "content": answer}
                )

                # Rerun to refresh UI with new messages
                st.rerun()
    else:
        st.info("📤 Upload at least one file to enable chat.")

with main_tab3:
    # -----------------------------
    # STORY SEARCH & FILTER SECTION
    # -----------------------------
    st.header("🔍 Story Search & Filter")
    st.markdown("**Search for similar stories from the CSV database using natural language queries. Results are ranked by tags and similarity.**")
    
    # Load stories from CSV
    stories = load_stories_from_csv()
    
    if not stories:
        st.info("📤 Upload story files first to enable search functionality. Stories are saved to CSV database automatically.")
    else:
        # Natural language query input
        search_query = st.text_area(
            "Enter your story idea, storyline, or description:",
            placeholder="e.g., 'A young detective solving a murder mystery in a small town' or 'Romantic comedy about two rivals falling in love'",
            height=100,
            key="story_search_query"
        )
        
        # Search button
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search Similar Stories", type="primary", key="search_stories")
        
        if search_button and search_query.strip():
            with st.spinner("🔍 Searching for similar stories and matching titles from database..."):
                try:
                    # Get existing titles from database that match the story idea
                    existing_titles = suggest_existing_titles_from_db(search_query, top_n=5)
                    
                    # Get similarity results from CSV database
                    results = find_similar_stories(search_query)
                    
                    # Display existing titles from database
                    if existing_titles:
                        st.markdown("### 📚 Suggested Titles")
                        st.markdown("**Here are actual story titles from uploaded files in the database that match your story idea:**")
                        st.info("💡 These are real titles from stories that have been analyzed and stored in the database.")
                        
                        for idx, title_data in enumerate(existing_titles, 1):
                            story_title = title_data.get('title', 'Unknown')
                            filename = title_data.get('filename', '')
                            display_title = story_title if story_title != filename else story_title
                            
                            with st.expander(f"**{idx}. {display_title}** (Relevance: {title_data['relevance']:.1f}%)", expanded=(idx == 1)):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**📄 Story Title:** {display_title}")
                                    if filename and filename != display_title:
                                        st.caption(f"📁 File: {filename}")
                                    st.write(f"**🎭 Genre:** {title_data.get('genre', 'unknown')}")
                                    st.write(f"**📊 Format:** {title_data.get('format', 'unknown').replace('_', ' ').title()}")
                                with col2:
                                    st.metric("Relevance", f"{title_data['relevance']:.1f}%")
                                
                                if title_data.get('themes'):
                                    st.write(f"**🏷️ Themes:** {title_data.get('themes', 'N/A')}")
                                
                                st.write(f"**📝 Summary:** {title_data.get('summary', 'N/A')}")
                                
                                if title_data.get('reasoning') and title_data.get('reasoning') != "Matched based on keywords":
                                    with st.expander("💭 Why this title matches your idea"):
                                        st.write(title_data.get('reasoning', ''))
                        
                        st.markdown("---")
                    
                    if results:
                        st.success(f"✅ Found {len(results)} similar story/stories (Ranked by Tags & Similarity)")
                        st.markdown("---")
                        
                        # Display ranked results pipeline
                        st.markdown("### 📊 Ranking Pipeline Results")
                        
                        for idx, result in enumerate(results, 1):
                            with st.expander(f"**Rank #{idx}: {result['title']}** - {result['similarity']:.1f}% similar", expanded=(idx == 1)):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**📄 Title:** {result['title']}")
                                    st.markdown(f"**📊 Format:** {result['format']}")
                                    st.markdown(f"**🎭 Genre:** {result['genre']}")
                                with col2:
                                    st.metric("Similarity", f"{result['similarity']:.1f}%")
                                
                                # Show tags
                                if result.get('tags'):
                                    st.markdown("**🏷️ Tags:**")
                                    tags_display = " ".join([f"`{tag}`" for tag in result['tags'] if tag.strip()])
                                    st.markdown(tags_display if tags_display else "No tags")
                                
                                st.markdown("**📝 Summary:**")
                                st.write(result['summary'])
                                
                                if result.get('characters'):
                                    st.markdown("**👥 Main Characters:**")
                                    for char in result['characters'][:5]:  # Show top 5
                                        if char and char.strip():
                                            st.write(f"- {char}")
                                
                                st.markdown("**🎯 Why it's similar:**")
                                st.info(result['reasoning'])
                                
                                if result.get('similarities'):
                                    st.markdown("**✅ Key Similarities:**")
                                    for sim in result['similarities'][:3]:
                                        st.write(f"• {sim}")
                    else:
                        st.warning("⚠️ No similar stories found. Try a different search query.")
                        
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
                    st.info("💡 Please check your API key and try again.")
        
        elif search_button:
            st.warning("⚠️ Please enter a story idea or description to search.")
        
        # Show database stats from CSV (simplified)
        st.markdown("---")
        total_stories = len(stories)
        st.info(f"📁 **Total stories in database:** {total_stories}")
        
        # List all story titles from CSV
        if stories:
            with st.expander("📋 View All Story Titles"):
                for idx, story in enumerate(stories, 1):
                    title = story.get("title", f"Story {idx}")
                    format_type = story.get("format", "unknown")
                    genre = story.get("genre", "unknown")
                    tags = story.get("themes", "")
                    st.write(f"{idx}. **{title}** ({format_type.replace('_', ' ').title()} - {genre})")
                    if tags:
                        st.caption(f"   Tags: {tags}")
