import os
import sys
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ---------------------------------------------------------------
# GEMINI CLIENT SETUP
# ---------------------------------------------------------------

# Load environment variables from .env
load_dotenv()

# Get the Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")

# Get the model name
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.6-flash")

# Make sure the API key was found
if not API_KEY:
    print("ERROR: GEMINI_API_KEY was not found in .env")
    print('Create a .env file with: GEMINI_API_KEY="your-key-here"')
    sys.exit(1)

# Create the Gemini client
client = genai.Client(api_key=API_KEY)

print("Gemini client setup successful!")


# ---------------------------------------------------------------
# NURU PRODUCT LINE MAP
# ---------------------------------------------------------------

NURU_PRODUCT_LINES = """
- Nuru Watoto: children 0-12 (bought by parents). Gentle, eczema-safe
  cleansing/moisturising. Hero products: fragrance-light cleanser,
  ceramide barrier balm, sun stick.
- Nuru Fresh: teens & Gen Z (~13-28). Acne management, oil control.
  Hero products: salicylic acid cleanser, oil-control gel, spot patch,
  mineral SPF.
- Nuru Even: millennials & working adults (~29-44). Post-acne
  hyperpigmentation, barrier repair, sun protection. Hero products:
  vitamin C serum, azelaic-acid brightening cream, daily SPF 50.
- Nuru Man: men, all ages. Oil control, razor-bump relief, sun
  protection. Hero products: oil-control face wash, razor-bump relief
  balm, all-in-one moisturiser+SPF.
- Nuru Mature: 55+. Medical-grade anti-ageing, sun-damage repair.
  Hero products: rich barrier repair cream, sun-damage serum, gentle
  cleansing balm.
- Nuru Roots: natural/afro haircare for women overall. Scalp oils,
  curl creams, protein treatments.
- If feedback doesn't clearly match a line (e.g. delivery, checkout,
  general brand comment), use "General / Not product-specific".
"""


# ---------------------------------------------------------------
# STAGE 1 — ANALYSE THE FEEDBACK (R-T-C-C-O PROMPT)
# ---------------------------------------------------------------

def build_analysis_prompt(feedback_text: str) -> tuple[str, str]:
    """
    Builds the Stage 1 prompt pair using the R-T-C-C-O framework.

    Returns:
        system_prompt: Role section
        user_prompt: Task, Context, Constraints and Output sections
    """

    # R - Role
    system_prompt = (
        "You are a senior Customer Insight Analyst for Nuru, a Kenyan "
        "clinical & dermo-cosmetic skincare and haircare brand. You "
        "specialise in reading raw customer feedback and extracting "
        "precise, structured insight, including which Nuru product "
        "line the feedback relates to."
    )

    # T - Task
    # C - Context
    # C - Constraints
    # O - Output
    user_prompt = f"""
# T - Task
Analyse the customer feedback provided below and extract structured
insight about sentiment, themes, urgency, and which Nuru product line
it relates to.

# C - Context
This feedback was submitted by a real Nuru customer. It may be a
product review, a complaint, or general feedback about the brand.

Nuru's product lines (use this to identify product_line_match):
{NURU_PRODUCT_LINES}

Feedback to analyse:
\"\"\"{feedback_text}\"\"\"

# C - Constraints
- Respond with VALID JSON ONLY. No commentary, no markdown fences.
- overall_sentiment must be one of: "positive", "negative", "neutral", "mixed"
- sentiment_score must be a number between -1.0 and 1.0
- urgency_level must be one of: "low", "medium", "high"
- product_line_match must be one of:
  "Nuru Watoto",
  "Nuru Fresh",
  "Nuru Even",
  "Nuru Man",
  "Nuru Mature",
  "Nuru Roots",
  "General / Not product-specific"
- key_themes and products_mentioned must be arrays of short strings.
  Use an empty array if none apply.

# O - Output format
Return exactly this JSON shape:
{{
  "overall_sentiment": "string",
  "sentiment_score": 0.0,
  "key_themes": ["string"],
  "products_mentioned": ["string"],
  "product_line_match": "string",
  "urgency_level": "string",
  "summary": "one sentence summary of the feedback"
}}
"""

    return system_prompt, user_prompt


# ---------------------------------------------------------------
# CORE API CALL FUNCTION (NEW GOOGLE GENAI CLIENT)
# ---------------------------------------------------------------

def call_ai(system_prompt: str, user_prompt: str) -> str:
    """
    Sends one request to the Gemini API using the Google GenAI SDK.
    Returns the raw text response.
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json"
        )
    )

    return response.text


# ---------------------------------------------------------------
# JSON PARSING
# ---------------------------------------------------------------

def parse_json_safely(raw_text: str) -> dict:
    """
    Safely parses Gemini's JSON response into a Python dictionary.
    """

    cleaned = raw_text.strip()

    # Handle accidental markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")

        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    return json.loads(cleaned)


# ---------------------------------------------------------------
# ANALYSE CUSTOMER FEEDBACK
# ---------------------------------------------------------------

def analyse_feedback(feedback_text: str) -> dict:
    """
    Runs the complete Stage 1 analysis:
    1. Builds the R-T-C-C-O prompt
    2. Sends it to Gemini
    3. Parses the JSON response
    """

    system_prompt, user_prompt = build_analysis_prompt(feedback_text)

    raw_response = call_ai(
        system_prompt,
        user_prompt
    )

    return parse_json_safely(raw_response)


# ---------------------------------------------------------------
# SIMPLE TEST
# ---------------------------------------------------------------

if __name__ == "__main__":

    feedback = input("\nEnter customer feedback: ")

    try:
        result = analyse_feedback(feedback)

        print("\n--- CUSTOMER FEEDBACK ANALYSIS ---")
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError:
        print("\nERROR: Gemini returned invalid JSON.")

    except Exception as error:
        print(f"\nERROR: {error}")