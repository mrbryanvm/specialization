"""
test_lead_scoring.py — Simulate 3 types of leads and verify scoring
=====================================================================

This script sends messages to the FastAPI /test endpoint to simulate
three different customer types and checks if the Lead Scoring system
correctly classifies them:

  1. HOT lead  — Carlos wants a specific product, has budget, urgent timeline
  2. WARM lead — María asks about products and prices but no commitment
  3. COLD lead — Someone just browsing, no real interest

REQUIREMENTS:
  - The server must be running: uvicorn server:app --reload --port 8000
  - The ChromaDB must be built: python chroma_helper.py

HOW TO RUN:
  python test_lead_scoring.py
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"


def send_message(message: str, description: str = "") -> dict:
    """Send a message to the /test endpoint and return the response."""
    print(f"\n{'─' * 60}")
    if description:
        print(f"📝 {description}")
    print(f"👤 CLIENTE: {message}")

    response = requests.get(f"{BASE_URL}/test", params={"q": message})
    data = response.json()

    print(f"🤖 ANA: {data['ai_response'][:200]}")

    if isinstance(data.get("lead_data"), dict):
        ld = data["lead_data"]
        emoji = {"HOT": "🔥", "WARM": "🟡", "COLD": "🔵"}.get(
            ld.get("score_label", ""), "❓"
        )
        print(f"\n   {emoji} Score: {ld.get('score', '?')}/10 ({ld.get('score_label', '?')})")
        print(f"   📋 Nombre: {ld.get('nombre', '-')}")
        print(f"   💰 Presupuesto: {ld.get('presupuesto', '-')}")
        print(f"   🛋️ Producto: {ld.get('producto_interes', '-')}")
        print(f"   ⏰ Timeline: {ld.get('timeline', '-')}")
        print(f"   📝 Resumen: {ld.get('resumen', '-')}")
    else:
        print(f"   ℹ️ Lead data: {data.get('lead_data', 'N/A')}")

    return data


def reset_test_conversation():
    """Clear the test conversation to start fresh."""
    try:
        requests.delete(f"{BASE_URL}/reset/test_user_browser")
        print("🔄 Test conversation reset.")
    except Exception:
        print("⚠️ Could not reset conversation (server may not be running).")


def test_hot_lead():
    """
    Simulate a HOT lead: Carlos wants a specific product, has a budget,
    and needs it urgently.
    Expected score: 8-10 (HOT)
    """
    print("\n" + "=" * 60)
    print("🔥 TEST 1: HOT LEAD (Carlos — ready to buy)")
    print("=" * 60)

    reset_test_conversation()
    time.sleep(1)

    send_message(
        "Hola, buenas tardes! Me llamo Carlos Mendoza.",
        "Step 1: Greeting + shares name (+1 pt)"
    )
    time.sleep(2)

    send_message(
        "Estoy buscando una mesa de comedor de madera para 6 personas.",
        "Step 2: Specific product interest (+2 pts)"
    )
    time.sleep(2)

    send_message(
        "Mi presupuesto es de unos 4000 Bs, ¿qué opciones tienen?",
        "Step 3: Budget specified (+2 pts) + within range (+1 pt)"
    )
    time.sleep(2)

    final = send_message(
        "La necesito antes del sábado si es posible. ¿Puedo ir al showroom mañana?",
        "Step 4: Timeline (+2 pts) + visit request (+2 pts) → should be HOT"
    )

    # Verify
    lead = final.get("lead_data", {})
    if isinstance(lead, dict):
        score = lead.get("score", 0)
        label = lead.get("score_label", "")
        print(f"\n{'─' * 60}")
        if label == "HOT" and score >= 8:
            print(f"✅ TEST PASSED! Score: {score}/10, Label: {label}")
        elif label == "HOT":
            print(f"⚠️ TEST PARTIAL: Label is HOT but score is {score}/10")
        else:
            print(f"❌ TEST FAILED: Expected HOT (8-10), got {label} ({score}/10)")
    else:
        print("❌ TEST FAILED: No structured lead data returned")


def test_warm_lead():
    """
    Simulate a WARM lead: María asks about products and prices
    but doesn't commit.
    Expected score: 5-7 (WARM)
    """
    print("\n" + "=" * 60)
    print("🟡 TEST 2: WARM LEAD (María — interested but browsing)")
    print("=" * 60)

    reset_test_conversation()
    time.sleep(1)

    send_message(
        "Hola, ¿qué mesas de comedor tienen?",
        "Step 1: General product interest"
    )
    time.sleep(2)

    send_message(
        "¿Y cuánto cuestan más o menos? Estoy buscando algo no muy caro.",
        "Step 2: Asks about prices, vague budget"
    )
    time.sleep(2)

    final = send_message(
        "Mmm interesante, voy a pensarlo. ¿Tienen catálogo que me puedan mandar?",
        "Step 3: No urgency, still considering"
    )

    # Verify
    lead = final.get("lead_data", {})
    if isinstance(lead, dict):
        score = lead.get("score", 0)
        label = lead.get("score_label", "")
        print(f"\n{'─' * 60}")
        if label == "WARM" and 5 <= score <= 7:
            print(f"✅ TEST PASSED! Score: {score}/10, Label: {label}")
        elif label in ("WARM", "COLD") and 3 <= score <= 7:
            print(f"⚠️ TEST ACCEPTABLE: Score: {score}/10, Label: {label}")
        else:
            print(f"❌ TEST UNEXPECTED: Expected WARM (5-7), got {label} ({score}/10)")
    else:
        print("❌ TEST FAILED: No structured lead data returned")


def test_cold_lead():
    """
    Simulate a COLD lead: Someone just browsing, minimal engagement.
    Expected score: 1-4 (COLD)
    """
    print("\n" + "=" * 60)
    print("🔵 TEST 3: COLD LEAD (Just browsing)")
    print("=" * 60)

    reset_test_conversation()
    time.sleep(1)

    send_message(
        "Hola",
        "Step 1: Just a greeting"
    )
    time.sleep(2)

    final = send_message(
        "Solo estoy viendo, gracias.",
        "Step 2: No interest expressed"
    )

    # Verify
    lead = final.get("lead_data", {})
    if isinstance(lead, dict):
        score = lead.get("score", 0)
        label = lead.get("score_label", "")
        print(f"\n{'─' * 60}")
        if label == "COLD" and score <= 4:
            print(f"✅ TEST PASSED! Score: {score}/10, Label: {label}")
        elif score <= 5:
            print(f"⚠️ TEST ACCEPTABLE: Score: {score}/10, Label: {label}")
        else:
            print(f"❌ TEST UNEXPECTED: Expected COLD (1-4), got {label} ({score}/10)")
    else:
        print("❌ TEST FAILED: No structured lead data returned")


def show_final_dashboard():
    """Show the leads dashboard after all tests."""
    print("\n" + "=" * 60)
    print("📊 FINAL LEADS DASHBOARD")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/leads")
    data = response.json()

    print(f"\nTotal leads tracked: {data['total_leads']}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("🧪 LEAD SCORING TEST SUITE — Day 14")
    print("Make sure the server is running: uvicorn server:app --port 8000")
    print()

    # Check if server is running
    try:
        health = requests.get(BASE_URL)
        print(f"Server status: {health.json().get('status', 'Unknown')}")
        print(f"Version: {health.json().get('version', 'Unknown')}")
    except requests.ConnectionError:
        print("❌ Server is not running! Start it with:")
        print("   uvicorn server:app --reload --port 8000")
        exit(1)

    # Run all three test scenarios
    test_hot_lead()
    time.sleep(3)

    test_warm_lead()
    time.sleep(3)

    test_cold_lead()
    time.sleep(2)

    # Show final dashboard
    show_final_dashboard()

    print("\n" + "=" * 60)
    print("🏁 ALL TESTS COMPLETE")
    print("=" * 60)
