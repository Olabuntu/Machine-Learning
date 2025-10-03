"""
Fine-tuning Project Demo Script
"""
import os
import sys

# Add src to path
sys.path.append('src')

from demo import main

if __name__ == "__main__":
    print("🎯 Fine-tuning Project")
    print("Professional Project Showcase")
    print("=" * 60)
    
    # Check if running in demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("🚀 Running in demo mode (no actual training)")
        main()
    else:
        print("💡 To run the demo: python run_demo.py --demo")
        print("💡 To run actual training: python run_training.py")
        print("\n📋 This project demonstrates:")
        print("  ✅ Advanced ML techniques (LoRA, quantization)")
        print("  ✅ Production-ready software architecture")
        print("  ✅ Comprehensive evaluation and monitoring")
        print("  ✅ Professional code quality and documentation")
        print("  ✅ Performance optimization and scalability")
