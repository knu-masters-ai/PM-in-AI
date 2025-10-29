# api/mailer.py
import os, json, smtplib
from typing import Optional, Dict, Any
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USERNAME or "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "1") == "1"


def _build_body_text(label: str, message: str, explanation: Optional[Dict[str, Any]]) -> str:
    lines = []
    lines.append(f"Result: {label}")
    if message:
        lines.append(message)
    if explanation:
        if summary := explanation.get("summary_text"):
            lines.append("\nSummary:")
            lines.append(summary)
        findings = explanation.get("findings") or []
        if findings:
            lines.append("\nFindings:")
            for i, f in enumerate(findings, 1):
                region = f.get("region", "(unknown)")
                lines.append(f"  #{i}: {region}")
                if f.get("evidence"):
                    lines.append(f"     note: {f['evidence']}")
        steps = explanation.get("next_steps") or []
        if steps:
            lines.append("\nNext steps:")
            for s in steps:
                lines.append(f"  - {s}")
        note = explanation.get("clinical_note")
        if note:
            lines.append("\nClinical note (for physician):")
            lines.append(note)
    lines.append("\n— KidneyStoneAI")
    return "\n".join(lines)


def send_results_email(
        to_email: str,
        label: str,
        message: str,
        original_bytes: bytes,
        original_filename: str,
        annotated_png_bytes: bytes,
        explanation: Optional[Dict[str, Any]] = None,
) -> None:
    if not (SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and SMTP_FROM):
        raise RuntimeError("SMTP is not configured (check SMTP_* env vars).")

    msg = MIMEMultipart()
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = "KidneyStoneAI: detection results"

    # Текст листа
    body_text = _build_body_text(label, message, explanation)
    msg.attach(MIMEText(body_text, "plain", "utf-8"))

    # Вкладення: оригінал
    part_orig = MIMEApplication(original_bytes, Name=original_filename)
    part_orig.add_header("Content-Disposition", "attachment", filename=original_filename)
    msg.attach(part_orig)

    # Вкладення: анотоване PNG
    part_ann = MIMEApplication(annotated_png_bytes, Name="annotated.png")
    part_ann.add_header("Content-Disposition", "attachment", filename="annotated.png")
    msg.attach(part_ann)

    # Вкладення: пояснення JSON (для зберігання)
    if explanation:
        exp_bytes = json.dumps(explanation, ensure_ascii=False, indent=2).encode("utf-8")
        part_json = MIMEApplication(exp_bytes, Name="explanation.json")
        part_json.add_header("Content-Disposition", "attachment", filename="explanation.json")
        msg.attach(part_json)

    # Відправка
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
        if SMTP_USE_TLS:
            s.starttls()
        s.login(SMTP_USERNAME, SMTP_PASSWORD)
        s.send_message(msg)
