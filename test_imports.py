#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")
try:
    from app.core.server import app
    from app.core.chains import answer_financial_question
    print("✅ All imports successful")
    # Try calling the router to check for issues
    from app.core.router import classify_query
    result = classify_query("What is the revenue for Q1?")
    print(f"✅ Router works: {result}")
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()
