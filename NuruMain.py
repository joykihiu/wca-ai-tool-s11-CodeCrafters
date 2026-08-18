# ---------------------------------------------------------------
# NURU PRODUCT LINE MAP (shared context for the analysis prompt)
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
    Builds the Stage 1 prompt pair (system, user) using R-T-C-C-O.
    Returns structured sentiment/theme/urgency/product-line data as JSON.
    """
    system_prompt = (
        # R - Role
        "You are a senior Customer Insight Analyst for Nuru, a Kenyan "
        "clinical & dermo-cosmetic skincare and haircare brand. You "
        "specialise in reading raw customer feedback and extracting "
        "precise, structured insight, including which Nuru product "
        "line the feedback relates to."
    )

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
- product_line_match must be one of: "Nuru Watoto", "Nuru Fresh",
  "Nuru Even", "Nuru Man", "Nuru Mature", "Nuru Roots",
  "General / Not product-specific"
- key_themes and products_mentioned must be arrays of short strings
  (use an empty array if none apply)

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