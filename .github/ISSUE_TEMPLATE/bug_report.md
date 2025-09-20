---
name: Bug Report
about: Report a bug or issue with the CAD Fusion Lab
title: '[BUG] '
labels: bug
assignees: ''

---

## 🐛 Bug Report

### Bug Description
A clear and concise description of the bug.

### Expected Behavior
What should happen?

### Actual Behavior
What actually happens?

### Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Run command '...'
4. See error

### Environment Information
- OS: [e.g. Ubuntu 20.04, Windows 10, macOS 12.0]
- Python Version: [e.g. 3.9.7]
- PyTorch Version: [e.g. 1.12.1]
- GPU: [e.g. NVIDIA RTX 3080, None]
- Installation Method: [e.g. pip, conda, source]

### Error Logs
```
Paste any relevant error messages or logs here
```

### Additional Context
- Screenshots (if applicable)
- Configuration files
- Sample data that triggers the bug
- Any workarounds you've found

### Suggested Fix (Optional)
If you have an idea about how to fix the bug, describe it here.

---

## 🔍 Recently Fixed Bugs (Reference)

### ✅ Critical Bugs Fixed in Recent Refactoring

1. **Dtype Mismatch in CAD Decoder**
   - **Issue**: RuntimeError: mat1 and mat2 must have the same dtype (Long vs Float)
   - **Solution**: Added proper type handling for both discrete tokens and continuous features
   - **Status**: ✅ Fixed

2. **Network Dependency in Tests**
   - **Issue**: Tests failing due to HuggingFace model downloads in offline environments
   - **Solution**: Added offline model support with mock encoders and tokenizers
   - **Status**: ✅ Fixed

3. **Geometric Validation Failures**
   - **Issue**: KCL generation and validation failing with parameter errors
   - **Solution**: Fixed parameter handling and added proper mock validation
   - **Status**: ✅ Fixed

4. **Import Errors with Missing Dependencies**
   - **Issue**: ModuleNotFoundError for optional dependencies like protobuf
   - **Solution**: Added graceful fallbacks and offline compatibility
   - **Status**: ✅ Fixed

5. **Memory Management Issues**
   - **Issue**: Potential memory leaks in data processing pipeline
   - **Solution**: Improved resource management and proper cleanup
   - **Status**: ✅ Fixed

6. **Configuration Management Problems**
   - **Issue**: Hardcoded values and no centralized configuration
   - **Solution**: Comprehensive configuration management system
   - **Status**: ✅ Fixed

7. **Error Handling Gaps**
   - **Issue**: Poor error handling leading to unhelpful error messages
   - **Solution**: Comprehensive exception handling with proper logging
   - **Status**: ✅ Fixed

8. **Test Infrastructure Issues**
   - **Issue**: Tests failing due to external dependencies and network calls
   - **Solution**: Complete test infrastructure overhaul with proper mocking
   - **Status**: ✅ Fixed

### 🎯 Common Bug Categories to Watch For

1. **Model Loading Issues**
   - Network connectivity problems
   - Missing model files
   - Version compatibility

2. **Data Processing Bugs**
   - Invalid input formats
   - Memory overflow with large datasets
   - Encoding/decoding errors

3. **Validation Failures**
   - Geometric constraint violations
   - Manufacturing feasibility issues
   - Format conversion errors

4. **Performance Issues**
   - Slow inference times
   - Memory leaks during training
   - GPU utilization problems

5. **Configuration Errors**
   - Invalid parameter values
   - Missing configuration files
   - Environment setup issues