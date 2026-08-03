"""
golden_set.py

12 hand-written job email/posting scenarios with known-correct expected
outputs, for eval_harness.py to run through the real parse -> score ->
plan pipeline (real Gemini/Groq calls, not mocked).

Each case's `expected` dict supports:
  - company_contains / role_contains: substring match, case-insensitive
  - expect_unresolved_company: True if the company should NOT resolve to a
    real name (tests the parser's honest failure mode, not a happy path)
  - fit_score_min / fit_score_max: acceptable range (LLM-driven parsing
    means exact-score assertions would be flaky; a range is the honest bar)
  - fit_score_informational: if True, an out-of-range score is reported but
    doesn't fail the case -- used for genuinely ambiguous/borderline inputs
    where there's no single "correct" score
  - expected_needs_followup / needs_followup_informational: same pattern
    for the Planner's decision

Case 4 deliberately targets the known bug (parse_node/nodes.py emitting a
literal "None"/"Not mentioned" company string when none is stated) -- this
golden set documents that behavior rather than avoiding it, per explicit
instruction not to dodge it.
"""

GOLDEN_SET = [
    {
        "id": "high_fit_clear_company",
        "raw_input": (
            "Hi, we're reaching out from Google. We have an opening for a "
            "Software Engineer, AI/ML focus. Required skills: Python, "
            "TensorFlow, scikit-learn, NumPy, pandas. Requires a Bachelor's "
            "degree in Computer Science. Minimum 3 years of experience."
        ),
        "expected": {
            "company_contains": "Google",
            "role_contains": "Software Engineer",
            "fit_score_min": 60.0,
            "fit_score_max": 100.0,
            "expected_needs_followup": True,
        },
    },
    {
        "id": "low_fit_unrelated_domain",
        "raw_input": (
            "We are hiring a Senior Marine Biologist with 15 years of "
            "deep-sea coral reef research experience. PhD in Marine "
            "Biology required. Skills: scuba certification, sample "
            "analysis, field research."
        ),
        "expected": {
            "fit_score_min": 0.0,
            "fit_score_max": 30.0,
            "expected_needs_followup": False,
        },
    },
    {
        "id": "borderline_near_threshold",
        "raw_input": (
            "Opening at Initech for a Machine Learning Engineer. Skills: "
            "TensorFlow, some Python. Requires 6+ years experience "
            "(negotiable). Bachelor's in any Engineering discipline."
        ),
        "expected": {
            "fit_score_min": 40.0,
            "fit_score_max": 80.0,
            "fit_score_informational": True,
            "expected_needs_followup": True,
            "needs_followup_informational": True,
        },
    },
    {
        "id": "no_company_mentioned",
        "raw_input": (
            "We have an exciting opportunity for a Data Scientist. Skills: "
            "Python, Machine Learning, pandas. 2+ years of experience "
            "preferred."
        ),
        "expected": {
            "expect_unresolved_company": True,
        },
    },
    {
        "id": "company_with_suffix_punctuation",
        "raw_input": (
            "Acme Corp., Inc. is looking for a Data Engineer. Required: "
            "Python, SQL, pandas, ETL pipeline experience. Bachelor's "
            "degree required. 3+ years experience."
        ),
        "expected": {
            "company_contains": "Acme",
            "role_contains": "Data Engineer",
        },
    },
    {
        "id": "perfect_skill_match",
        "raw_input": (
            "TechNova Inc. AI Engineer role. Required skills: TensorFlow, "
            "Keras, OpenCV, scikit-learn, NumPy, pandas. Bachelor's degree "
            "in Computer Science or related field. 4+ years experience."
        ),
        "expected": {
            "company_contains": "TechNova",
            "fit_score_min": 75.0,
            "fit_score_max": 100.0,
            "expected_needs_followup": True,
        },
    },
    {
        "id": "education_mismatch_only",
        "raw_input": (
            "BrightPath Analytics seeks an ML Engineer. Skills: TensorFlow, "
            "scikit-learn, pandas. PhD in Computer Science required. 4 "
            "years experience."
        ),
        "expected": {
            "company_contains": "BrightPath",
            "fit_score_min": 30.0,
            "fit_score_max": 90.0,
            "fit_score_informational": True,
        },
    },
    {
        "id": "years_mismatch_only",
        "raw_input": (
            "Orion Systems Senior AI Architect. Required: TensorFlow, "
            "Keras, scikit-learn, pandas expertise. Bachelor's in Computer "
            "Science. Minimum 12 years of experience required."
        ),
        "expected": {
            "company_contains": "Orion",
            "fit_score_min": 20.0,
            "fit_score_max": 75.0,
            "fit_score_informational": True,
        },
    },
    {
        "id": "verbose_noisy_email",
        "raw_input": (
            "Dear Candidate,\n\nThank you for your interest in career "
            "opportunities. Our organization has been recognized for "
            "workplace excellence for the past decade running, and we take "
            "great pride in our inclusive culture and commitment to "
            "innovation across all our global offices spanning multiple "
            "continents.\n\nCompany: Redwood Analytics\nRole: Machine "
            "Learning Engineer\nWe have followed your profile with interest "
            "and would like to discuss the following opportunity in "
            "detail.\n\nRequired Skills: Python, TensorFlow, pandas, "
            "NumPy\nExperience: 3+ years\nEducation: Bachelor's degree in "
            "Computer Science or related discipline\n\nWe look forward to "
            "hearing from you and hope to schedule a conversation at your "
            "earliest convenience. Please let us know your availability.\n\n"
            "Warm regards,\nTalent Acquisition Team"
        ),
        "expected": {
            "company_contains": "Redwood",
            "role_contains": "Machine Learning",
            "fit_score_min": 50.0,
            "fit_score_max": 100.0,
        },
    },
    {
        "id": "malformed_garbage_input",
        "raw_input": "asdkjaskjd 123 !!! xyz ### foobar",
        "expected": {
            "expect_unresolved_company": True,
            "fit_score_informational": True,
            "fit_score_min": 0.0,
            "fit_score_max": 100.0,
        },
    },
    {
        "id": "genai_stack_high_fit",
        "raw_input": (
            "Cascade AI Labs, GenAI Engineer role. Required: LangChain, "
            "RAG, embeddings, HuggingFace, Python. Bachelor's degree in "
            "Computer Science. 3+ years experience with LLM applications."
        ),
        "expected": {
            "company_contains": "Cascade",
            "fit_score_min": 40.0,
            "fit_score_max": 100.0,
        },
    },
    {
        "id": "interview_invite_style",
        "raw_input": (
            "Hi, we'd like to invite you for an interview for the Software "
            "Engineer role at Meridian Systems."
        ),
        "expected": {
            "company_contains": "Meridian",
            "role_contains": "Software Engineer",
        },
    },
]
