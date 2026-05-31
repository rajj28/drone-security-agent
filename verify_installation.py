"""Verify BLIP and optimization installation"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

print("=" * 70)
print("🔍 VERIFICATION REPORT")
print("=" * 70)

# Check 1: Core dependencies
print("\n📦 Core Dependencies:")
try:
    import torch
    print(f"  ✅ torch {torch.__version__}")
except:
    print("  ❌ torch not found")

try:
    import transformers
    print(f"  ✅ transformers {transformers.__version__}")
except:
    print("  ❌ transformers not found")

try:
    import timm
    print(f"  ✅ timm {timm.__version__}")
except:
    print("  ❌ timm not found")

# Check 2: Performance libraries
print("\n⚡ Performance Libraries:")
try:
    import orjson
    print("  ✅ orjson (fast JSON)")
except:
    print("  ⚠️  orjson not found (optional)")

try:
    import lru
    print("  ✅ lru-dict (fast cache)")
except:
    print("  ⚠️  lru-dict not found (optional)")

try:
    import psutil
    print(f"  ✅ psutil {psutil.__version__} (system monitoring)")
except:
    print("  ❌ psutil not found")

# Check 3: BLIP availability
print("\n🤖 BLIP Models:")
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    print("  ✅ BLIP Processor & Model available")
except Exception as e:
    print(f"  ⚠️  BLIP not ready: {e}")

try:
    from transformers import BlipForQuestionAnswering
    print("  ✅ BLIP VQA Model available")
except:
    print("  ⚠️  BLIP VQA not available")

# Check 4: Custom modules
print("\n📝 Custom Modules:")
try:
    from optimization_utils import MemoryOptimizer
    print("  ✅ optimization_utils")
except Exception as e:
    print(f"  ❌ optimization_utils: {e}")

try:
    from blip_vision_analyzer import OptimizedBLIPVisionAnalyzer
    print("  ✅ blip_vision_analyzer")
except Exception as e:
    print(f"  ❌ blip_vision_analyzer: {e}")

try:
    from unified_vision_analyzer import UnifiedVisionAnalyzer
    print("  ✅ unified_vision_analyzer")
except Exception as e:
    print(f"  ❌ unified_vision_analyzer: {e}")

# Check 5: System optimization
print("\n🚀 System Optimization:")
try:
    from optimization_utils import quick_optimize
    config = quick_optimize()
    print(f"  ✅ Auto-configuration working")
except Exception as e:
    print(f"  ⚠️  Auto-config: {e}")

print("\n" + "=" * 70)
print("✅ VERIFICATION COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("  1. Run demo: python demo_blip_optimizer.py")
print("  2. Read guide: INTEGRATE_BLIP.md")
print("  3. Integrate into your pipeline")
