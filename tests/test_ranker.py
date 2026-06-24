"""
Automated tests for ranking and compliance verification.
"""
import os
import json
import pytest
import time
from validate_submission import validate_submission
from src.scoring import score_candidate
from src.honeypot import check_is_honeypot
from src.reasoning import generate_reasoning_string

# Paths
SUBMISSION_PATH = r"D:\projects\redrob\submission.csv"

def test_trap_vs_fit_scoring():
    """
    Ensures 'Marketing Manager with AI keywords' trap scores extremely low (0)
    and 'Plain-language builder' (no trendy keywords but shipped search system at product co) scores high.
    """
    # 1. Marketing Manager Trap Profile
    trap_candidate = {
        "candidate_id": "CAND_9999901",
        "profile": {
            "anonymized_name": "Trap Candidate",
            "headline": "Marketing Manager | Expert in RAG, Pinecone, and LLM Fine-tuning",
            "summary": "Experienced marketing leader specializing in AI buzzwords.",
            "location": "Noida, India",
            "country": "India",
            "years_of_experience": 8.0,
            "current_title": "Marketing Manager",
            "current_company": "Acme Corp",
            "current_company_size": "201-500",
            "current_industry": "Marketing"
        },
        "career_history": [
            {
                "company": "Acme Corp",
                "title": "Marketing Manager",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 53,
                "is_current": True,
                "industry": "Marketing",
                "company_size": "201-500",
                "description": "Led marketing strategy and user acquisition campaigns. Talked about RAG and LLMs on LinkedIn."
            }
        ],
        "education": [
            {
                "institution": "Delhi University",
                "degree": "BBA",
                "field_of_study": "Marketing",
                "start_year": 2014,
                "end_year": 2017,
                "tier": "tier_2"
            }
        ],
        "skills": [
            {"name": "rag", "proficiency": "expert", "endorsements": 10, "duration_months": 24},
            {"name": "pinecone", "proficiency": "expert", "endorsements": 15, "duration_months": 24},
            {"name": "llms", "proficiency": "expert", "endorsements": 8, "duration_months": 12}
        ],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "signup_date": "2022-01-01",
            "last_active_date": "2026-06-15",
            "open_to_work_flag": True,
            "profile_views_received_30d": 12,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.9,
            "avg_response_time_hours": 4.0,
            "skill_assessment_scores": {},
            "connection_count": 150,
            "endorsements_received": 25,
            "notice_period_days": 15,
            "expected_salary_range_inr_lpa": {"min": 15.0, "max": 25.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": -1,
            "search_appearance_30d": 45,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 1.0,
            "offer_acceptance_rate": 1.0,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }
    
    # 2. Plain-language Builder Fit Profile
    fit_candidate = {
        "candidate_id": "CAND_9999902",
        "profile": {
            "anonymized_name": "Fit Candidate",
            "headline": "Lead Software Engineer | Applied Machine Learning",
            "summary": "Backend and systems engineer focusing on information retrieval and model deployment.",
            "location": "Pune, India",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior Software Engineer",
            "current_company": "Razorpay",
            "current_company_size": "1001-5000",
            "current_industry": "Fintech"
        },
        "career_history": [
            {
                "company": "Razorpay",
                "title": "Senior Software Engineer",
                "start_date": "2021-06-20",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "Fintech",
                "company_size": "1001-5000",
                "description": "Architected and deployed our internal search and recommendation API. Handled indices of 5M products, using FAISS vector indexing, indexing drift, and NDCG evaluation offline to validate retrieval quality improvements."
            }
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2015,
                "end_year": 2019,
                "tier": "tier_1"
            }
        ],
        "skills": [
            {"name": "python", "proficiency": "expert", "endorsements": 85, "duration_months": 72},
            {"name": "docker", "proficiency": "advanced", "endorsements": 40, "duration_months": 36},
            {"name": "git", "proficiency": "advanced", "endorsements": 30, "duration_months": 48}
        ],
        "redrob_signals": {
            "profile_completeness_score": 95.0,
            "signup_date": "2020-01-01",
            "last_active_date": "2026-06-19",
            "open_to_work_flag": True,
            "profile_views_received_30d": 80,
            "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.95,
            "avg_response_time_hours": 2.0,
            "skill_assessment_scores": {"Python": 92.0},
            "connection_count": 450,
            "endorsements_received": 120,
            "notice_period_days": 15,
            "expected_salary_range_inr_lpa": {"min": 25.0, "max": 35.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 85,
            "search_appearance_30d": 120,
            "saved_by_recruiters_30d": 25,
            "interview_completion_rate": 1.0,
            "offer_acceptance_rate": 1.0,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }
    
    # Run scoring
    trap_score, _, is_trap_disq, _ = score_candidate(trap_candidate, semantic_score=0.45)
    fit_score, fit_breakdown, is_fit_disq, _ = score_candidate(fit_candidate, semantic_score=0.68)
    
    # Assert trap scores 0 (is disqualified due to title-substance mismatch)
    assert is_trap_disq == True
    assert trap_score == 0.0
    
    # Assert fit candidate passes and scores high
    assert is_fit_disq == False
    assert fit_score > 0.6
    # Verify semantic match score pulls weight (norm_semantic should be high)
    assert fit_breakdown["norm_semantic"] >= 0.6

def test_honeypot_detection():
    """
    Verifies that the honeypot detection flags artificial contradictions.
    """
    honeypot_candidate = {
        "candidate_id": "CAND_HP00001",
        "profile": {
            "anonymized_name": "Honeypot Candidate",
            "headline": "Senior Engineer",
            "years_of_experience": 2.0,
            "current_title": "Software Engineer",
            "current_company": "Initech"
        },
        "career_history": [
            {
                "company": "Initech",
                "title": "Software Engineer",
                "start_date": "2023-01-01",
                "end_date": "2025-01-01",
                "duration_months": 120, # IMPOSSIBLE: 10 years in a 2 year date range!
                "is_current": False,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Coding."
            }
        ],
        "education": [],
        "skills": [],
        "redrob_signals": {}
    }
    
    is_hp, reasons = check_is_honeypot(honeypot_candidate)
    assert is_hp == True
    assert any("Career Date Mismatch" in r for r in reasons)

def test_reasoning_generation_checks():
    """
    Verifies reasoning string properties: specific facts, connection to JD, honest concerns, rank consistency.
    """
    candidate = {
        "candidate_id": "CAND_9999902",
        "profile": {
            "anonymized_name": "Fit Candidate",
            "years_of_experience": 7.0,
            "current_title": "Senior Software Engineer"
        },
        "skills": [
            {"name": "python", "proficiency": "expert", "duration_months": 72}
        ],
        "career_history": [
            {"company": "Razorpay", "title": "Senior Software Engineer", "duration_months": 60}
        ],
        "redrob_signals": {
            "notice_period_days": 90, # long notice
            "recruiter_response_rate": 0.95
        }
    }
    
    breakdown = {
        "career_breakdown": {"all_services": False, "is_title_chaser": False},
        "behavioral_breakdown": {"days_since_active": 10, "behavior_multiplier": 0.8}
    }
    
    # Generate reasoning for rank 5 (Top Pick)
    reasoning_top = generate_reasoning_string(candidate, rank=5, score=0.85, breakdown=breakdown)
    # Generate reasoning for rank 95 (Marginal Pick)
    reasoning_bottom = generate_reasoning_string(candidate, rank=95, score=0.45, breakdown=breakdown)
    
    # 1. Fact checking
    assert "7.0" in reasoning_top or "7.0" in reasoning_bottom  # Specific fact: YOE
    assert "Razorpay" in reasoning_top or "Razorpay" in reasoning_bottom  # Specific fact: Company
    
    # 2. Honest concern checking (90 days notice should trigger notice concern)
    assert "90 days" in reasoning_top or "90 days" in reasoning_bottom
    
    # 3. Tone consistency checking
    assert "Founding Team fit" in reasoning_top or "shipper" in reasoning_top
    assert "filler" in reasoning_bottom or "Marginal" in reasoning_bottom
    
    # 4. Variation check
    assert reasoning_top != reasoning_bottom

def test_validator_passes():
    """
    Verifies the output submission file matches the hackathon validation script perfectly.
    """
    if os.path.exists(SUBMISSION_PATH):
        errors = validate_submission(SUBMISSION_PATH)
        assert len(errors) == 0, f"Validator found errors: {errors}"
    else:
        pytest.skip("submission.csv does not exist yet. Run rank.py to produce it.")
