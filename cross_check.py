"""
Cross-check documentation against actual code
"""
import sys
import torch
sys.path.insert(0, '.')

from models.echoflow_v2 import EchoFlowV2, count_parameters

print("="*70)
print("CROSS-CHECK: DOCUMENTATION vs CODE")
print("="*70)

errors = []
warnings = []

# 1. Check actual parameter counts
print("\n[1/5] Verifying parameter counts...")
model = EchoFlowV2()
params = count_parameters(model)

actual_total = params['total_millions']
actual_trainable = params['trainable_millions']
actual_frozen = params['frozen_millions']

print(f"Actual total: {actual_total:.2f}M")
print(f"Actual trainable: {actual_trainable:.2f}M")
print(f"Actual frozen: {actual_frozen:.2f}M")

# Check README
readme = open('README.md').read()
if '338' not in readme or '338.13' not in readme:
    errors.append("README doesn't mention correct total params (338.13M)")
if '22' not in readme and '23' not in readme and '22.70' not in readme:
    errors.append("README doesn't mention correct trainable params (22.70M)")

# 2. Check training time claims
print("\n[2/5] Verifying training time claims...")
if '10-12' in readme or '10-12 часов' in readme:
    print("✓ Training time claim found: 10-12 hours")
else:
    warnings.append("Training time claim not found in README")

# 3. Check accuracy claims
print("\n[3/5] Verifying accuracy claims...")
if '96' in readme or '97' in readme or '98' in readme or '99' in readme:
    print("✓ Accuracy claims found")
else:
    warnings.append("Accuracy claims not clear in README")

# 4. Verify model architecture components
print("\n[4/5] Verifying architecture components...")
model_code = open('models/echoflow_v2.py').read()

required_components = {
    'Wav2Vec2': 'Wav2Vec2Model',
    'TransformerEncoder': 'class TransformerEncoder',
    'SqueezeExcitation': 'class SqueezeExcitation',
    'StochasticDepth': 'class StochasticDepth',
    'MultiScaleFeatureFusion': 'class MultiScaleFeatureFusion',
    'AdvancedAttentionPooling': 'class AdvancedAttentionPooling'
}

for name, code_pattern in required_components.items():
    if code_pattern in model_code:
        print(f"✓ {name} found")
    else:
        errors.append(f"Component {name} not found (pattern: {code_pattern})")

# 5. Check train.py CLI arguments
print("\n[5/5] Verifying CLI arguments...")
train_code = open('train.py').read()

cli_args = ['--data_dir', '--batch_size', '--epochs', '--lr', '--weight_decay']
for arg in cli_args:
    if arg in train_code:
        print(f"✓ {arg} found")
    else:
        warnings.append(f"CLI argument {arg} not found")

# Summary
print("\n" + "="*70)
print("CROSS-CHECK SUMMARY")
print("="*70)

if errors:
    print(f"\n❌ ERRORS: {len(errors)}")
    for error in errors:
        print(f"  ❌ {error}")
else:
    print("\n✅ NO ERRORS")

if warnings:
    print(f"\n⚠️  WARNINGS: {len(warnings)}")
    for warning in warnings:
        print(f"  ⚠️  {warning}")
else:
    print("\n✅ NO WARNINGS")

print("\n" + "="*70)

if errors:
    print("STATUS: ❌ DOCUMENTATION MISMATCH")
    sys.exit(1)
elif warnings:
    print("STATUS: ⚠️  MINOR ISSUES")
    sys.exit(0)
else:
    print("STATUS: ✅ PERFECT MATCH")
    sys.exit(0)
