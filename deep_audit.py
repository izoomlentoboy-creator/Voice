"""
Deep Audit Script for EchoFlow 2.0
Checks for errors, inaccuracies, contradictions
"""

import sys
import os
import re
from pathlib import Path

print("="*70)
print("ECHOFLOW 2.0 - DEEP AUDIT")
print("="*70)

errors = []
warnings = []
info = []

# 1. Check file structure
print("\n[1/10] Checking file structure...")
required_files = [
    'README.md',
    'requirements.txt',
    'train.py',
    'train_ultimate.sh',
    'models/echoflow_v2.py',
    'utils/dataset.py',
    'utils/augmentation.py',
    'test_integration.py'
]

for file in required_files:
    if not Path(file).exists():
        errors.append(f"Missing required file: {file}")
    else:
        info.append(f"✓ Found: {file}")

print(f"✓ Checked {len(required_files)} required files")

# 2. Check parameter count consistency
print("\n[2/10] Checking parameter count consistency...")
param_mentions = {}

files_to_check = ['README.md', 'models/echoflow_v2.py', 'train.py', 'OPTIMIZATION_AUDIT.md']
for file in files_to_check:
    if Path(file).exists():
        content = Path(file).read_text()
        
        # Look for parameter mentions
        total_matches = re.findall(r'(\d+)M?\s+(?:total\s+)?param', content, re.IGNORECASE)
        trainable_matches = re.findall(r'(\d+)M?\s+trainable', content, re.IGNORECASE)
        
        if total_matches or trainable_matches:
            param_mentions[file] = {
                'total': total_matches,
                'trainable': trainable_matches
            }

# Check consistency
expected_total = ['338', '338.13']
expected_trainable = ['22', '23', '22.70']

for file, params in param_mentions.items():
    for total in params.get('total', []):
        if total not in expected_total and not any(exp in total for exp in expected_total):
            warnings.append(f"{file}: Unexpected total param count: {total}M (expected ~338M)")
    
    for trainable in params.get('trainable', []):
        if trainable not in expected_trainable and not any(exp in trainable for exp in expected_trainable):
            warnings.append(f"{file}: Unexpected trainable param count: {trainable}M (expected ~23M)")

print(f"✓ Checked parameter consistency across {len(param_mentions)} files")

# 3. Check for deprecated code patterns
print("\n[3/10] Checking for deprecated patterns...")
py_files = list(Path('.').rglob('*.py'))

deprecated_patterns = [
    (r'torch\.cuda\.amp\.autocast\(', 'Use torch.amp.autocast with try/except for compatibility'),
]

for py_file in py_files:
    if 'test_' in str(py_file) or '__pycache__' in str(py_file):
        continue
    
    content = py_file.read_text()
    for pattern, suggestion in deprecated_patterns:
        if re.search(pattern, content):
            # Check if there's a try/except around it
            if 'try:' not in content or 'torch.amp.autocast' not in content:
                warnings.append(f"{py_file}: {suggestion}")

print(f"✓ Checked {len(py_files)} Python files for deprecated patterns")

# 4. Check import consistency
print("\n[4/10] Checking import consistency...")
try:
    sys.path.insert(0, '.')
    from models.echoflow_v2 import EchoFlowV2, count_parameters
    from utils.dataset import create_dataloaders
    from utils.augmentation import AugmentationPipeline
    print("✓ All imports successful")
except Exception as e:
    errors.append(f"Import error: {e}")

# 5. Check documentation consistency
print("\n[5/10] Checking documentation consistency...")
readme = Path('README.md').read_text()

# Check for contradictions
if '330M' in readme:
    errors.append("README still mentions old 330M parameter count")
if '18M trainable' in readme or '18M обучаемых' in readme:
    errors.append("README still mentions old 18M trainable count")

# Check for required sections
required_sections = ['Быстрый старт', 'Архитектура', 'Оптимизации', 'Требования']
for section in required_sections:
    if section not in readme:
        warnings.append(f"README missing section: {section}")

print("✓ Documentation consistency checked")

# 6. Check requirements.txt
print("\n[6/10] Checking requirements.txt...")
requirements = Path('requirements.txt').read_text()
required_packages = ['torch', 'transformers', 'librosa', 'numpy', 'scikit-learn']

for package in required_packages:
    if package not in requirements:
        errors.append(f"Missing package in requirements.txt: {package}")

print(f"✓ Checked {len(required_packages)} required packages")

# 7. Check train_ultimate.sh
print("\n[7/10] Checking train_ultimate.sh...")
if Path('train_ultimate.sh').exists():
    script = Path('train_ultimate.sh').read_text()
    
    # Check for required sections
    if 'python' not in script.lower() and 'python3' not in script.lower():
        errors.append("train_ultimate.sh doesn't call Python")
    
    if 'train.py' not in script:
        errors.append("train_ultimate.sh doesn't call train.py")
    
    print("✓ train_ultimate.sh structure checked")

# 8. Check model architecture consistency
print("\n[8/10] Checking model architecture...")
model_file = Path('models/echoflow_v2.py').read_text()

# Check for key components
required_components = [
    'Wav2Vec2',
    'TransformerEncoder',
    'SqueezeExcitation',
    'StochasticDepth',
    'MultiScaleFeatureFusion',
    'AdvancedAttentionPooling'
]

for component in required_components:
    if component not in model_file:
        warnings.append(f"Model might be missing component: {component}")

print(f"✓ Checked for {len(required_components)} key components")

# 9. Check for TODO/FIXME/XXX comments
print("\n[9/10] Checking for unresolved TODOs...")
for py_file in py_files:
    content = py_file.read_text()
    for marker in ['TODO', 'FIXME', 'XXX', 'HACK']:
        if marker in content:
            warnings.append(f"{py_file}: Contains {marker} comment")

print("✓ Checked for unresolved markers")

# 10. Check for hardcoded values that should be configurable
print("\n[10/10] Checking for hardcoded values...")
train_file = Path('train.py').read_text()

# These should be in argparse
configurable_params = ['batch_size', 'learning_rate', 'epochs', 'num_workers']
for param in configurable_params:
    if f'--{param}' not in train_file and f'--{param.replace("_", "-")}' not in train_file:
        warnings.append(f"train.py: {param} might not be configurable via CLI")

print("✓ Checked for hardcoded values")

# Summary
print("\n" + "="*70)
print("AUDIT SUMMARY")
print("="*70)

if errors:
    print(f"\n❌ ERRORS FOUND: {len(errors)}")
    for error in errors:
        print(f"  ❌ {error}")
else:
    print("\n✅ NO ERRORS FOUND")

if warnings:
    print(f"\n⚠️  WARNINGS: {len(warnings)}")
    for warning in warnings[:10]:  # Show first 10
        print(f"  ⚠️  {warning}")
    if len(warnings) > 10:
        print(f"  ... and {len(warnings) - 10} more warnings")
else:
    print("\n✅ NO WARNINGS")

print(f"\nℹ️  INFO: {len(info)} items checked")

print("\n" + "="*70)

if errors:
    print("STATUS: ❌ ISSUES FOUND - NEEDS FIXES")
    sys.exit(1)
elif warnings:
    print("STATUS: ⚠️  WARNINGS FOUND - REVIEW RECOMMENDED")
    sys.exit(0)
else:
    print("STATUS: ✅ PERFECT - NO ISSUES")
    sys.exit(0)
