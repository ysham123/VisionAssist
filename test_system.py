#!/usr/bin/env python3
"""
VisionAssist System Test
Simple test script to validate core functionality
"""
import requests
import base64
import json
import time
from io import BytesIO
from PIL import Image, ImageDraw

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 250], fill='white', outline='black', width=2)
    draw.text((200, 150), "TEST IMAGE", fill='black', anchor='mm')
    
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode()

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get('http://127.0.0.1:5000/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def test_vision_caption():
    """Test vision caption endpoint"""
    try:
        image_data = create_test_image()
        response = requests.post(
            'http://127.0.0.1:5000/api/v1/vision/caption',
            json={'image': f'data:image/jpeg;base64,{image_data}'},
            timeout=30
        )
        return response.status_code == 200 and 'caption' in response.json()
    except:
        return False

def test_conversation():
    """Test conversation endpoint"""
    try:
        response = requests.post(
            'http://127.0.0.1:5000/api/v1/conversation/chat',
            json={'message': 'Hello, can you help me?'},
            timeout=10
        )
        return response.status_code == 200 and 'response' in response.json()
    except:
        return False

def main():
    """Run all tests"""
    print("🧪 VisionAssist System Test")
    print("=" * 40)
    
    tests = [
        ("Health Check", test_health),
        ("Vision Caption", test_vision_caption),
        ("Conversation", test_conversation)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Testing {name}...", end=" ")
        result = test_func()
        status = "✅ PASS" if result else "❌ FAIL"
        print(status)
        results.append(result)
    
    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Check server logs.")

if __name__ == '__main__':
    main()
