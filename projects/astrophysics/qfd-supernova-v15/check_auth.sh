#!/bin/bash
# Check Claude Code authentication and usage status

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Claude Code Authentication Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check credentials file
if [ -f ~/.claude/.credentials.json ]; then
    echo "✓ Credentials file found: ~/.claude/.credentials.json"

    # Try to extract email (if available)
    if command -v jq &> /dev/null; then
        EMAIL=$(jq -r '.user_email // "Not found"' ~/.claude/.credentials.json 2>/dev/null)
        echo "  Account email: $EMAIL"
    else
        echo "  (Install jq to see account details: sudo apt install jq)"
    fi
else
    echo "✗ No credentials file found"
    echo "  Run: claude setup-token"
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""

# Check Claude Code version
echo "Claude Code Version:"
claude --version

echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""

# Show diagnostic info
echo "Diagnostic Info:"
claude doctor

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Next Steps for Research Preview Authorization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "If you're seeing weekly limit warnings:"
echo ""
echo "1. Re-authenticate with your Research Preview account:"
echo "   $ claude setup-token"
echo ""
echo "2. Check your usage limits in the Anthropic Console:"
echo "   https://console.anthropic.com/settings/limits"
echo ""
echo "3. Verify your Research Preview enrollment:"
echo "   - Look for $1000 credit balance"
echo "   - Check that tier shows 'Research Preview' or similar"
echo "   - Weekly limits should be significantly higher or removed"
echo ""
echo "4. If still limited, contact Anthropic Support:"
echo "   Email: support@anthropic.com"
echo "   Subject: Claude Code Research Preview - Rate Limit Issue"
echo "   Include: Your account email and Research Preview enrollment confirmation"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
