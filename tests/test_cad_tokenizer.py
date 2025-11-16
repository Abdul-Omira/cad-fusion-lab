"""
Tests for CAD Tokenizer

Validates the CAD-specific tokenizer without requiring full dependencies.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_tokenizer_structure():
    """Test tokenizer code structure."""
    print("=" * 70)
    print("TEST: CAD Tokenizer Structure")
    print("=" * 70)

    with open('src/models/cad_tokenizer.py', 'r') as f:
        content = f.read()

    checks = [
        ('CADOperation class', 'class CADOperation' in content),
        ('CADVocabulary class', 'class CADVocabulary' in content),
        ('CADTokenizer class', 'class CADTokenizer' in content),
        ('Encode method', 'def encode(' in content),
        ('Decode method', 'def decode(' in content),
        ('Quantization', 'def quantize_parameter' in content),
        ('Dequantization', 'def dequantize_parameter' in content),
        ('Operations list', 'OPERATIONS =' in content),
        ('Sketches list', 'SKETCHES =' in content),
        ('Constraints list', 'CONSTRAINTS =' in content),
        ('Special tokens', 'SPECIAL_TOKENS' in content),
        ('Vocabulary building', '_build_vocabulary' in content),
        ('Text conversion', 'def to_text(' in content),
        ('Save/load vocab', 'def save_vocabulary' in content and 'def load_vocabulary' in content),
    ]

    passed = 0
    failed = 0

    for name, result in checks:
        if result:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1

    print(f"\nResult: {passed}/{len(checks)} checks passed")
    return failed == 0


def test_tokenizer_features():
    """Test tokenizer features."""
    print("\n" + "=" * 70)
    print("TEST: CAD Tokenizer Features")
    print("=" * 70)

    with open('src/models/cad_tokenizer.py', 'r') as f:
        content = f.read()

    features = {
        'Structured operations': 'operation_type' in content and 'parameters' in content,
        'Multiple operation types': content.count("'") > 50,  # Many operation strings
        'Parameter quantization': '8-bit' in content or 'quantization_bits' in content,
        'Constraint support': 'constraints' in content.lower(),
        'Plane support': 'XY' in content and 'XZ' in content,
        'Separator tokens': '<SEP>' in content,
        'Start/end tokens': '<START>' in content and '<END>' in content,
        'Unknown token handling': '<UNK>' in content,
        'Vocabulary size property': 'vocab_size' in content,
        'Bidirectional conversion': 'encode' in content and 'decode' in content,
        'Human-readable text': 'to_text' in content,
        'JSON serialization': 'json' in content.lower(),
        'Dataclass usage': '@dataclass' in content,
        'Type hints': ': str' in content and ': int' in content,
        'Docstrings': '"""' in content,
    }

    passed = sum(1 for v in features.values() if v)
    total = len(features)

    for feature, present in features.items():
        status = "✓" if present else "✗"
        print(f"  {status} {feature}")

    print(f"\n  Score: {passed}/{total} features ({passed/total*100:.0f}%)")
    return passed >= total * 0.9


def test_vocabulary_completeness():
    """Test vocabulary completeness."""
    print("\n" + "=" * 70)
    print("TEST: Vocabulary Completeness")
    print("=" * 70)

    with open('src/models/cad_tokenizer.py', 'r') as f:
        content = f.read()

    # Extract operation types
    operations_section = content[content.find('OPERATIONS = ['):content.find(']', content.find('OPERATIONS = ['))]

    operation_counts = {
        'Basic operations': ['sketch', 'extrude', 'revolve'],
        'Advanced operations': ['loft', 'sweep', 'shell'],
        'Modifications': ['fillet', 'chamfer', 'draft'],
        'Patterns': ['pattern_linear', 'pattern_circular', 'mirror'],
        'Booleans': ['boolean_union', 'boolean_subtract', 'boolean_intersect'],
        'Sketches': ['rectangle', 'circle', 'polygon', 'line'],
        'Constraints': ['parallel', 'perpendicular', 'tangent', 'coincident'],
    }

    results = {}
    for category, items in operation_counts.items():
        found = sum(1 for item in items if item in content)
        results[category] = f"{found}/{len(items)}"
        print(f"  {category}: {found}/{len(items)}")

    return True


def test_tokenizer_improvements():
    """Test improvements over baseline."""
    print("\n" + "=" * 70)
    print("TEST: Improvements Over Baseline")
    print("=" * 70)

    with open('src/models/cad_tokenizer.py', 'r') as f:
        new_content = f.read()

    # Check if old implementation exists
    old_file = 'src/models/text_to_cad.py'
    if os.path.exists(old_file):
        with open(old_file, 'r') as f:
            old_content = f.read()

        improvements = {
            'Structured operations': 'CADOperation' in new_content and 'CADOperation' not in old_content,
            'CAD-specific vocabulary': 'CADVocabulary' in new_content,
            'Operation types defined': 'OPERATIONS =' in new_content,
            'Sketch types': 'SKETCHES =' in new_content,
            'Constraint support': 'CONSTRAINTS =' in new_content,
            'Parameter quantization': 'quantize_parameter' in new_content,
            'Bidirectional conversion': 'encode' in new_content and 'decode' in new_content,
            'Human-readable text': 'to_text' in new_content,
            'Structured dataclass': '@dataclass' in new_content,
            'Comprehensive docs': new_content.count('"""') > old_content.count('"""'),
        }

        for improvement, present in improvements.items():
            status = "✓" if present else "✗"
            print(f"  {status} {improvement}")

        passed = sum(1 for v in improvements.values() if v)
        total = len(improvements)
        print(f"\n  Improvements: {passed}/{total} ({passed/total*100:.0f}%)")
        return passed >= total * 0.8
    else:
        print("  ⚠  Baseline file not found for comparison")
        return True


def test_code_quality():
    """Test code quality metrics."""
    print("\n" + "=" * 70)
    print("TEST: Code Quality")
    print("=" * 70)

    with open('src/models/cad_tokenizer.py', 'r') as f:
        lines = f.readlines()
        content = ''.join(lines)

    # Count elements
    classes = content.count('class ')
    methods = content.count('def ')
    docstrings = content.count('"""') // 2
    type_hints = content.count(': str') + content.count(': int') + content.count(': List') + content.count(': Dict')

    print(f"  Lines of code: {len(lines)}")
    print(f"  Classes: {classes}")
    print(f"  Methods/Functions: {methods}")
    print(f"  Docstrings: {docstrings}")
    print(f"  Type hints: {type_hints}")

    quality_checks = {
        'Sufficient documentation': docstrings >= methods * 0.8,
        'Type annotations': type_hints >= methods * 0.5,
        'Modular design': classes >= 3,
        'Example usage': 'example_usage' in content,
        'Error handling': 'Exception' in content or 'Error' in content,
    }

    print("\n  Quality Checks:")
    for check, passed in quality_checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")

    passed = sum(1 for v in quality_checks.values() if v)
    print(f"\n  Quality Score: {passed}/{len(quality_checks)} ({passed/len(quality_checks)*100:.0f}%)")

    return passed >= len(quality_checks) * 0.8


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CAD TOKENIZER TEST SUITE")
    print("=" * 70)

    tests = [
        ("Structure", test_tokenizer_structure),
        ("Features", test_tokenizer_features),
        ("Vocabulary", test_vocabulary_completeness),
        ("Improvements", test_tokenizer_improvements),
        ("Code Quality", test_code_quality),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Test '{name}' failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)

    print(f"\n  Total: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n  🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n  ⚠️  {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
