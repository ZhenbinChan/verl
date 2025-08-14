#!/usr/bin/env python3
"""
Simple vLLM server test script
"""

import requests
import json

def test_vllm_server(url="http://127.0.0.1:8000"):
    print(f"Testing vLLM server at: {url}")
    
    # 1. Check if server is running
    try:
        response = requests.get(f"{url}/health", timeout=5)
        print(f"✅ Server is running: {response.status_code}")
    except:
        print("❌ Server is not running!")
        return
    
    # 2. Get available models
    try:
        response = requests.get(f"{url}/v1/models", timeout=5)
        models = response.json()["data"]
        model_name = models[0]["id"]
        print(f"✅ Found model: {model_name}")
    except:
        print("❌ Cannot get models!")
        return
    
    # 3. Test simple chat
    try:
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 50
        }
        
        response = requests.post(
            f"{url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            print(f"✅ Chat works! Answer: {answer.strip()}")
        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Chat error: {e}")

if __name__ == "__main__":
    test_vllm_server()