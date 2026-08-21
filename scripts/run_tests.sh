#!/bin/bash
# scripts/run_tests.sh
# سكريبت الاختبارات

set -e

echo "🧪 Running tests..."

pytest tests/ -v --cov=mouthlocnet --cov-report=html

echo "✅ Tests complete!"
echo "📊 Coverage report: htmlcov/index.html"