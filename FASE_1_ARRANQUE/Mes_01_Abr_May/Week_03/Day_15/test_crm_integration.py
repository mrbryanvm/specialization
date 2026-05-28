"""
test_crm_integration.py — Test the Make.com webhook + CRM pipeline
====================================================================

This script tests the Day 15 CRM integration in 3 ways:

  1. WEBHOOK TEST: Sends a fake HOT lead directly to the /test-webhook
     endpoint, which pushes it to Make.com.
  2. CONVERSATION TEST: Simulates a full HOT lead conversation via /test
     endpoint and verifies the lead data reaches the webhook.
  3. COLD LEAD TEST: Simulates a COLD lead and verifies it does NOT
     get sent to the webhook (only WARM and HOT leads are sent).

REQUIREMENTS:
  - The server must be running: uvicorn server:app --port 8000
  - MAKE_WEBHOOK_URL must be configured in .env
  - The Make.com scenario must be active (toggle ON)

HOW TO RUN:
  python test_crm_integration.py
"""

import json
import time

import requests

BASE_URL = "http://localhost:8000"


def check_server():
    """Verify the server is running and webhook is configured."""
    print("=" * 60)
    print("🔍 CHECKING SERVER STATUS")
    print("=" * 60)

    try:
        response = requests.get(BASE_URL)
        data = response.json()

        print(f"  Status: {data.get('status', 'Unknown')}")
        print(f"  Version: {data.get('version', 'Unknown')}")
        print(f"  CRM Webhook: {data.get('crm_webhook', 'Unknown')}")

        if "NOT configured" in str(data.get("crm_webhook", "")):
            print("\n  ❌ MAKE_WEBHOOK_URL is not set in .env!")
            print("  Add your Make.com webhook URL to the .env file first.")
            return False

        print("\n  ✅ Server is ready for CRM testing!")
        return True

    except requests.ConnectionError:
        print("  ❌ Server is not running!")
        print("  Start it with: uvicorn server:app --port 8000")
        return False


def test_1_direct_webhook():
    """
    TEST 1: Send a fake lead directly to the /test-webhook endpoint.
    This is the quickest way to verify that Make.com receives data.
    """
    print("\n" + "=" * 60)
    print("🧪 TEST 1: DIRECT WEBHOOK TEST (fake HOT lead)")
    print("=" * 60)

    response = requests.post(f"{BASE_URL}/test-webhook")
    data = response.json()

    print(f"\n  Status: {data.get('status', 'Unknown')}")
    print(f"  HTTP Status: {data.get('http_status', 'Unknown')}")

    if "payload_sent" in data:
        payload = data["payload_sent"]
        print(f"\n  📦 Payload sent to Make.com:")
        print(f"     Nombre: {payload.get('nombre')}")
        print(f"     Teléfono: {payload.get('telefono')}")
        print(f"     Producto: {payload.get('producto_interes')}")
        print(f"     Presupuesto: {payload.get('presupuesto')}")
        print(f"     Score: {payload.get('score')}/10 ({payload.get('score_label')})")

    if data.get("next_steps"):
        print(f"\n  📋 Next steps:")
        for step in data["next_steps"]:
            print(f"     {step}")

    if data.get("http_status") == 200:
        print(f"\n  ✅ TEST 1 PASSED!")
    else:
        print(f"\n  ❌ TEST 1 FAILED — Check Make.com scenario status")

    return data


def test_2_hot_lead_conversation():
    """
    TEST 2: Simulate a full HOT lead conversation and verify CRM push.
    """
    print("\n" + "=" * 60)
    print("🔥 TEST 2: HOT LEAD CONVERSATION → CRM")
    print("=" * 60)

    # Reset the test conversation first
    requests.delete(f"{BASE_URL}/reset/test_user_browser")
    print("  🔄 Test conversation reset.")
    time.sleep(1)

    # Step 1: Greeting with name
    print("\n  Step 1: Greeting + name")
    r1 = requests.get(
        f"{BASE_URL}/test",
        params={"q": "Hola, me llamo María Fernández."},
    )
    d1 = r1.json()
    print(f"  🤖 ANA: {d1['ai_response'][:120]}...")
    print(f"  📊 Score: {d1['lead_data'].get('score', '?') if isinstance(d1['lead_data'], dict) else 'N/A'}")
    print(f"  📡 CRM: {d1.get('crm_status', 'N/A')}")
    time.sleep(3)

    # Step 2: Specific product + budget
    print("\n  Step 2: Product + Budget")
    r2 = requests.get(
        f"{BASE_URL}/test",
        params={"q": "Busco un sofá de cuero para la sala. Mi presupuesto es 6000 Bs."},
    )
    d2 = r2.json()
    print(f"  🤖 ANA: {d2['ai_response'][:120]}...")
    print(f"  📊 Score: {d2['lead_data'].get('score', '?') if isinstance(d2['lead_data'], dict) else 'N/A'}")
    print(f"  📡 CRM: {d2.get('crm_status', 'N/A')}")
    time.sleep(3)

    # Step 3: Timeline + visit request (should trigger HOT)
    print("\n  Step 3: Timeline + Visit request → should become HOT")
    r3 = requests.get(
        f"{BASE_URL}/test",
        params={
            "q": "Lo necesito antes del viernes. ¿Puedo pasar mañana por el showroom?"
        },
    )
    d3 = r3.json()
    print(f"  🤖 ANA: {d3['ai_response'][:120]}...")

    if isinstance(d3["lead_data"], dict):
        ld = d3["lead_data"]
        print(f"\n  📊 Final Score: {ld.get('score', '?')}/10 ({ld.get('score_label', '?')})")
        print(f"  📋 Nombre: {ld.get('nombre', '-')}")
        print(f"  💰 Presupuesto: {ld.get('presupuesto', '-')}")
        print(f"  🛋️ Producto: {ld.get('producto_interes', '-')}")
        print(f"  ⏰ Timeline: {ld.get('timeline', '-')}")
        print(f"  📝 Resumen: {ld.get('resumen', '-')}")

    print(f"  📡 CRM Status: {d3.get('crm_status', 'N/A')}")

    # Verify
    if isinstance(d3["lead_data"], dict):
        label = d3["lead_data"].get("score_label", "")
        if label == "HOT":
            print(f"\n  ✅ TEST 2 PASSED! Lead is HOT and should be in Google Sheets + email sent!")
        elif label == "WARM":
            print(f"\n  ⚠️ TEST 2 PARTIAL: Lead is WARM — logged in Sheets but no email alert")
        else:
            print(f"\n  ❌ TEST 2 UNEXPECTED: Expected HOT, got {label}")


def test_3_cold_lead_no_crm():
    """
    TEST 3: Simulate a COLD lead and verify it does NOT get sent to CRM.
    """
    print("\n" + "=" * 60)
    print("🔵 TEST 3: COLD LEAD → should NOT be sent to CRM")
    print("=" * 60)

    # Reset
    requests.delete(f"{BASE_URL}/reset/test_user_browser")
    print("  🔄 Test conversation reset.")
    time.sleep(1)

    # Just a greeting
    print("\n  Step 1: Simple greeting")
    r1 = requests.get(f"{BASE_URL}/test", params={"q": "Hola"})
    d1 = r1.json()
    print(f"  🤖 ANA: {d1['ai_response'][:120]}...")
    print(f"  📡 CRM: {d1.get('crm_status', 'N/A')}")

    if d1.get("crm_status") == "Not sent to CRM yet":
        print(f"\n  ✅ TEST 3 PASSED! COLD lead was NOT sent to CRM (correct behavior)")
    else:
        print(f"\n  ⚠️ TEST 3: CRM status is '{d1.get('crm_status')}' — check server logs")


def show_final_status():
    """Show the final server status after all tests."""
    print("\n" + "=" * 60)
    print("📊 FINAL SERVER STATUS")
    print("=" * 60)

    response = requests.get(BASE_URL)
    data = response.json()

    print(f"\n  Leads tracked: {data.get('leads_tracked', 0)}")
    print(f"  Leads sent to CRM: {data.get('leads_sent_to_crm', 0)}")
    print(f"  Lead summary: {json.dumps(data.get('leads_summary', {}), indent=4)}")


if __name__ == "__main__":
    print("🧪 CRM INTEGRATION TEST SUITE — Day 15")
    print("Make sure the server is running: uvicorn server:app --port 8000")
    print("Make sure MAKE_WEBHOOK_URL is set in .env")
    print("Make sure your Make.com scenario is ON\n")

    if not check_server():
        exit(1)

    # Test 1: Direct webhook test
    test_1_direct_webhook()
    time.sleep(2)

    # Test 2: Full HOT lead conversation
    test_2_hot_lead_conversation()
    time.sleep(2)

    # Test 3: COLD lead should not reach CRM
    test_3_cold_lead_no_crm()

    # Final status
    show_final_status()

    print("\n" + "=" * 60)
    print("🏁 ALL CRM TESTS COMPLETE")
    print("=" * 60)
    print("\n📋 CHECKLIST — Verify manually:")
    print("   [ ] Google Sheet has new rows")
    print("   [ ] Email alert received for HOT lead")
    print("   [ ] No email for COLD lead")
