#!/usr/bin/env python3
"""
Debug vLLM server connection issues
"""

import requests
import json
import time
import subprocess
import socket

def check_port_open(host, port):
    """Check if port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def test_vllm_detailed(url="http://127.0.0.1:8000"):
    print(f"🔍 Detailed testing of vLLM server at: {url}")
    
    # Parse URL
    if "://" in url:
        protocol, rest = url.split("://", 1)
        if ":" in rest:
            host, port = rest.split(":", 1)
            port = int(port)
        else:
            host = rest
            port = 80 if protocol == "http" else 443
    else:
        host, port = "127.0.0.1", 8000
    
    print(f"Host: {host}, Port: {port}")
    
    # 1. Check if port is open
    print(f"\n1. Checking if port {port} is open...")
    if check_port_open(host, port):
        print(f"✅ Port {port} is open")
    else:
        print(f"❌ Port {port} is closed or not reachable")
        print("💡 Check if vLLM server is running")
        return
    
    # 2. Check basic HTTP connection
    print(f"\n2. Testing basic HTTP connection...")
    try:
        response = requests.get(f"{url}/", timeout=5)
        print(f"✅ Basic HTTP works: {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return
    except requests.exceptions.Timeout:
        print(f"❌ Connection timeout")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return
    
    # 3. Check health endpoint
    print(f"\n3. Testing health endpoint...")
    try:
        response = requests.get(f"{url}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        if response.status_code != 200:
            print("❌ Health check failed")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # 4. Check models endpoint
    print(f"\n4. Testing models endpoint...")
    try:
        response = requests.get(f"{url}/v1/models", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"Models response: {json.dumps(models, indent=2)}")
            if models.get("data"):
                model_name = models["data"][0]["id"]
                print(f"✅ Using model: {model_name}")
            else:
                print("❌ No models found")
                return
        else:
            print(f"❌ Models endpoint failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return
    
    # 5. Test chat completions with verbose output
    print(f"\n5. Testing chat completions...")
    try:
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        print(f"Request payload: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            f"{url}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            json=data,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat works!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"❌ Chat error: {e}")

def check_vllm_process():
    """Check if vLLM process is running"""
    print("\n🔍 Checking vLLM processes...")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        vllm_processes = [line for line in result.stdout.split('\n') if 'vllm' in line.lower()]
        
        if vllm_processes:
            print("✅ Found vLLM processes:")
            for proc in vllm_processes:
                print(f"  {proc}")
        else:
            print("❌ No vLLM processes found")
            
    except Exception as e:
        print(f"Error checking processes: {e}")

def check_gpu_usage():
    """Check GPU usage"""
    print("\n🔍 Checking GPU usage...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU info:")
            # Just show the process part
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'Processes:' in line:
                    for j in range(i, min(i+10, len(lines))):
                        if lines[j].strip():
                            print(f"  {lines[j]}")
                    break
        else:
            print("❌ Cannot get GPU info")
    except Exception as e:
        print(f"Error checking GPU: {e}")

if __name__ == "__main__":
    import sys
    
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    
    check_vllm_process()
    check_gpu_usage()
    test_vllm_detailed(url)