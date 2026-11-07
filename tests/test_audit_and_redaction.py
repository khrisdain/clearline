from clearline.audit import MemoryAuditLog
from clearline.auth import DEV_USERS
from clearline.redact import redact


def test_the_chain_verifies_and_detects_edits_and_deletions():
    log = MemoryAuditLog()
    for i in range(5):
        log.append("ana", "ask", {"question": f"q{i}"})
    assert log.verify() is None
    log._entries[2].detail["question"] = "something else"
    assert log.verify() == 3
    log = MemoryAuditLog()
    for i in range(4):
        log.append("ana", "ask", {"question": f"q{i}"})
    del log._entries[1]
    assert log.verify() == 3


def test_questions_are_logged_with_personal_information_masked(service):
    service.ask("Can I expense parking? My email is sam@lakeview.ca", DEV_USERS["ana"])
    entry = service.audit.entries(1)[0]
    assert entry.action == "ask" and entry.subject == "ana"
    assert "sam@lakeview.ca" not in entry.detail["question"]
    assert entry.detail["grounded"] in (True, False)


def test_redaction_checks_the_sin_digit_and_leaves_ordinary_numbers():
    assert redact("SIN 046-454-286") == "SIN [SIN]"
    assert redact("Order 123 456 789") == "Order 123 456 789"  # fails the Luhn check
    assert redact("Call (613) 555-0142 or +1 416 555 0199") == "Call [PHONE] or [PHONE]"
    assert redact("budget of $412.6 million in 2027") == "budget of $412.6 million in 2027"
