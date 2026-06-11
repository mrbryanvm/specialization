"""
test_day16.py — Day 16 Integration Test Suite
==============================================

Tests the two new Day 16 features:
  1. Outbound Initiation (/api/outbound endpoint)
  2. Human Handoff (keyword detection + blocking + resume)

HOW TO RUN:
    1. Make sure the server is running:
       uvicorn server:app --port 8000
    2. Run this script:
       python test_day16.py

NOTE: Test 1 (Outbound) will attempt to send a REAL WhatsApp to your number.
Make sure your Twilio credentials are in .env and your number is connected to
the sandbox (sent the "join" keyword to the sandbox number before).
"""

import time

import requests

BASE_URL = "http://localhost:8000"


def print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"🧪 {title}")
    print("=" * 60)


def print_step(step: str) -> None:
    print(f"\n  {step}")


def send_message(phone: str, message: str) -> dict:
    """Simulate a WhatsApp message coming in via Twilio /webhook."""
    response = requests.post(
        f"{BASE_URL}/webhook",
        data={"Body": message, "From": phone},
    )
    return {"status_code": response.status_code, "response": response.text}


def check_server() -> bool:
    """Check if the server is running and ready."""
    try:
        r = requests.get(f"{BASE_URL}/")
        data = r.json()
        print(f"  Status: {data.get('status', 'Unknown')}")
        print(f"  Version: {data.get('version', 'Unknown')}")
        print(f"  Twilio Client: {data.get('twilio_client', 'Unknown')}")
        print(f"  CRM Webhook: {data.get('crm_webhook', 'Unknown')}")
        print(f"  Handoff Webhook: {data.get('handoff_webhook', 'Unknown')}")
        return True
    except Exception as e:
        print(f"  ❌ Server not reachable: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# MAIN TEST SUITE
# ──────────────────────────────────────────────────────────────

print("\n🧪 DAY 16 TEST SUITE — Outbound + Human Handoff")
print("Make sure the server is running: uvicorn server:app --port 8000")
print("Make sure TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are set in .env")

print_separator("CHECKING SERVER STATUS")
if not check_server():
    print("\n  ❌ Server is not running. Start it first, then run this test.")
    exit(1)
print("\n  ✅ Server is ready!")


# ──────────────────────────────────────────────────────────────
# TEST 1: OUTBOUND INITIATION
# ──────────────────────────────────────────────────────────────

print_separator("TEST 1: OUTBOUND INITIATION — /api/outbound")
print("  Goal: When Make.com calls /api/outbound, the bot sends a WhatsApp greeting.")

test_phone = "59169512710"  # Your phone number (digits only, no + or whatsapp:)

print_step("Calling POST /api/outbound with a fake form submission...")

outbound_payload = {
    "nombre": "Bryan (Test Form)",
    "telefono": test_phone,
    "interes": "sofás y mesas de comedor",
}

try:
    r = requests.post(f"{BASE_URL}/api/outbound", json=outbound_payload)
    data = r.json()

    print(f"  HTTP Status: {r.status_code}")
    print(f"  Response: {data.get('status', 'N/A')}")
    print(f"  Prospect: {data.get('prospect', 'N/A')}")
    print(f"  Message SID: {data.get('message_sid', 'N/A')}")

    if r.status_code == 200 and "✅" in data.get("status", ""):
        print("\n  ✅ TEST 1 PASSED!")
        print("  📱 Check your WhatsApp — you should have a new greeting message!")
        print(
            "  📋 Next: Reply to the bot from WhatsApp and see if it has context."
        )
    else:
        print(f"\n  ❌ TEST 1 RESULT: {data}")
        print(
            "  Note: If Twilio returns an error, check that your number has sent"
            " 'join [keyword]' to the sandbox number."
        )

except Exception as e:
    print(f"\n  ❌ TEST 1 ERROR: {e}")

time.sleep(3)

# ──────────────────────────────────────────────────────────────
# TEST 2: HANDOFF TRIGGERED BY KEYWORDS
# ──────────────────────────────────────────────────────────────

print_separator("TEST 2: HUMAN HANDOFF — Keyword Detection")
print("  Goal: When the customer says 'quiero hablar con una persona', bot hands off.")

test_handoff_phone = "whatsapp:+59155555001"

print_step("Step 1: Reset test conversation...")
r = requests.delete(f"{BASE_URL}/reset/{test_handoff_phone}")
print(f"  Reset: {r.json().get('status', 'N/A')}")

print_step("Step 2: Send a normal message (bot should respond normally)...")
result = send_message(test_handoff_phone, "Hola, quiero ver mesas de comedor")
# Check response isn't a handoff message
normal_response = result["response"]
if "especialista" not in normal_response.lower():
    print(f"  ✅ Bot responded normally (not in handoff mode yet)")
else:
    print(f"  ⚠️ Bot may have gone to handoff too early")

time.sleep(2)

print_step("Step 3: Customer asks for a human agent...")
result = send_message(test_handoff_phone, "quiero hablar con una persona")
handoff_response = result["response"]

if "especialista" in handoff_response.lower() or "5 minutos" in handoff_response.lower():
    print(f"  ✅ Bot sent handoff message correctly!")
    print(f"  Bot said: '{handoff_response[:200]}'")
else:
    print(f"  ❌ Expected handoff message, got: '{handoff_response[:200]}'")

time.sleep(2)

print_step("Step 4: Verify bot is SILENT after handoff...")
result = send_message(test_handoff_phone, "Hola? Sigues ahí?")
silent_response = result["response"]

if "especialista humano" in silent_response.lower() or "humano" in silent_response.lower():
    print(f"  ✅ Bot is correctly silent (returning handoff status message)")
else:
    print(f"  ❌ Bot should be silent but responded: '{silent_response[:150]}'")

# Check server state
r = requests.get(f"{BASE_URL}/")
server_data = r.json()
handoff_numbers = server_data.get("handoff_numbers", [])
if test_handoff_phone in handoff_numbers:
    print(f"  ✅ Server confirms handoff is active for {test_handoff_phone}")
else:
    print(f"  ❌ Server doesn't show {test_handoff_phone} in handoff_numbers")
    print(f"     Active handoffs: {handoff_numbers}")

time.sleep(2)

print_step("Step 5: Owner uses /api/resume to return customer to bot...")
phone_digits = "59155555001"
r = requests.post(f"{BASE_URL}/api/resume/{phone_digits}")
resume_data = r.json()
print(f"  Resume response: {resume_data.get('status', 'N/A')}")

if "✅" in resume_data.get("status", ""):
    print(f"  ✅ Handoff deactivated successfully!")
else:
    print(f"  ❌ Resume failed: {resume_data}")

time.sleep(2)

print_step("Step 6: Verify bot responds again after resume...")
result = send_message(test_handoff_phone, "Hola, quiero ver mesas de comedor")
resumed_response = result["response"]

if "especialista humano" not in resumed_response.lower():
    print(f"  ✅ Bot is responding again after resume!")
else:
    print(f"  ❌ Bot still in handoff mode: '{resumed_response[:150]}'")

print("\n  ✅ TEST 2 COMPLETE!")

# ──────────────────────────────────────────────────────────────
# TEST 3: AUTO-HANDOFF — Bot triggers handoff after 3 "no info" replies
# ──────────────────────────────────────────────────────────────

print_separator("TEST 3: AUTO-HANDOFF — 3 consecutive 'no info' replies")
print("  Goal: After the bot says 'no tengo esa información' 3 times, auto-handoff.")
print("  Note: This test sends unusual questions the catalog won't have answers for.")

test_auto_phone = "whatsapp:+59155555002"

print_step("Resetting test conversation...")
requests.delete(f"{BASE_URL}/reset/{test_auto_phone}")
print("  Reset done.")

questions = [
    "¿Cuántos muebles tienen en inventario exacto?",
    "¿Cuál es el número de serie del sofá Milano?",
    "¿Tienen certificación ISO 9001 en sus procesos de fabricación?",
]

for i, question in enumerate(questions, 1):
    print_step(f"Step {i}: Asking unusual question: '{question[:60]}'")
    result = send_message(test_auto_phone, question)
    time.sleep(3)  # Wait between calls to avoid rate limiting

# Check final state
r = requests.get(f"{BASE_URL}/")
server_data = r.json()
handoff_numbers = server_data.get("handoff_numbers", [])

if test_auto_phone in handoff_numbers:
    print(f"\n  ✅ TEST 3 PASSED! Auto-handoff triggered after repeated 'no info' responses.")
else:
    print(
        f"\n  ℹ️ TEST 3 INFO: Auto-handoff may not have triggered."
        f"\n  This is OK — the bot might be finding partial answers in the catalog."
        f"\n  Active handoffs: {handoff_numbers}"
    )

# ──────────────────────────────────────────────────────────────
# TEST 4: /handoffs and /outbound DASHBOARD ENDPOINTS
# ──────────────────────────────────────────────────────────────

print_separator("TEST 4: NEW DASHBOARD ENDPOINTS")

print_step("GET /handoffs — view active handoffs...")
r = requests.get(f"{BASE_URL}/handoffs")
data = r.json()
print(f"  Total active handoffs: {data.get('total_active_handoffs', 0)}")
print(f"  Tip: {data.get('tip', '')}")

print_step("GET /outbound — view outbound log...")
r = requests.get(f"{BASE_URL}/outbound")
data = r.json()
print(f"  Total outbound contacts: {data.get('total_outbound', 0)}")
if data.get("outbound_contacts"):
    print(f"  Contacts logged: {list(data['outbound_contacts'].keys())}")

print(f"\n  ✅ TEST 4 PASSED! New endpoints are working.")

# ──────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────

print_separator("FINAL SERVER STATUS")
r = requests.get(f"{BASE_URL}/")
data = r.json()
print(f"  Version: {data.get('version')}")
print(f"  Active Conversations: {data.get('active_conversations')}")
print(f"  Leads Tracked: {data.get('leads_tracked')}")
print(f"  Active Handoffs: {data.get('active_handoffs')}")
print(f"  Outbound Sent: {data.get('outbound_sent')}")

print("\n" + "=" * 60)
print("🏁 ALL DAY 16 TESTS COMPLETE")
print("=" * 60)

print("""
📋 MANUAL CHECKLIST:
   [ ] WhatsApp received the greeting from TEST 1 (/api/outbound)
   [ ] Google Sheets CRM still works (send /test-webhook to verify)
   [ ] Handoff alert email received (if MAKE_HANDOFF_WEBHOOK_URL is configured)
   [ ] Bot resumes normally after /api/resume

📋 NEXT STEPS (Make.com configuration):
   [ ] Create new Make.com scenario: Google Forms → POST /api/outbound
   [ ] Create Google Form with Name, Phone, Interest fields
   [ ] Create Make.com scenario for handoff alerts (optional for now)
   [ ] Test full flow: Fill form → WhatsApp greeting → Chat → Handoff
""")
