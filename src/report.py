"""
Create a PDF report for a single email prediction.
"""

from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def build_pdf(result, subject, sender, reply_to):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER))

    story = [
        Paragraph("Phishing Email Detection Report", styles["CenterTitle"]),
        Spacer(1, 12),
        Paragraph(f"<b>Classification:</b> {result['label']}", styles["BodyText"]),
        Paragraph(f"<b>Risk level:</b> {result['risk_level']}", styles["BodyText"]),
        Paragraph(f"<b>Risk score:</b> {result['risk_score']:.2f}%", styles["BodyText"]),
        Spacer(1, 10),
        Paragraph(f"<b>Subject:</b> {subject or '(none)'}", styles["BodyText"]),
        Paragraph(f"<b>Sender:</b> {sender or '(none)'}", styles["BodyText"]),
        Paragraph(f"<b>Reply-To:</b> {reply_to or '(none)'}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Security indicators", styles["Heading2"])
    ]

    rows = [["Indicator", "Value"]]
    for k, v in result["features"].items():
        rows.append([k.replace("_", " ").title(), str(v)])
    table = Table(rows, colWidths=[280, 160])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#263238")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey])
    ]))
    story += [table, Spacer(1, 12), Paragraph(
        "This report is a probabilistic ML assessment and should not be treated as proof that an email is malicious.",
        styles["Italic"]
    )]

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
