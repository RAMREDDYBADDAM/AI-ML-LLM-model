#!/usr/bin/env python3
"""
Minimal test server to debug startup issues.
"""
import sys
import traceback

try:
    print("Step 1: Loading environment...")
    from app.config import settings
    print(f"  ✅ Settings loaded. OpenAI key: {'***' if settings.openai_api_key else 'MISSING'}")
    
    print("Step 2: Importing router...")
    from app.core.router import classify_query
    print("  ✅ Router imported")
    
    print("Step 3: Importing vectorstore (this calls OpenAI & Chroma)...")
    from app.core.vectorstore import get_vectorstore
    try:
        vs = get_vectorstore()
        print("  ✅ Vectorstore initialized")
    except Exception as e:
        print(f"  ⚠️ Vectorstore init failed (non-fatal): {e}")
    
    print("Step 4: Importing chains...")
    from app.core.chains import answer_financial_question
    print("  ✅ Chains imported")
    
    print("Step 5: Importing server app...")
    from app.core.server import app
    print("  ✅ Server app imported")
    
    print("\n✅ All startup checks passed!")
    
    # Try running the router
    print("\nStep 6: Testing router...")
    result = classify_query("Show me Apple revenue for Q1")
    print(f"  ✅ Router result: {result}")
    
except Exception as e:
    print(f"\n❌ Startup failed at one of the steps above.")
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)
